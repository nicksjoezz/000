"""
Live trading on Polymarket CLOB V2 via `polymarket-apis`.

Polymarket migrated to CLOB V2. New wallets trade through the gasless
**deposit-wallet** flow (signature_type=3 / POLY_1271):

  - Funds live in a **deposit wallet** derived from your EOA (deposit via Polymarket).
  - Your **private key / seed (EOA)** signs every order — it holds no funds or gas.
  - A **relayer API key** sponsors on-chain setup (wallet deploy + token approvals),
    so you never pay gas.

Legacy accounts (Polymarket proxy / Gnosis safe) are auto-detected and used as-is:
we pick whichever wallet actually holds pUSD. The EOA is derived from PRIVATE_KEY
(hex key or 12/24-word seed phrase), so there is no signature-type/funder to choose
by hand any more.

All clients are synchronous — call from the event loop via `asyncio.to_thread`.
"""

import threading
from typing import Optional, Dict, Any, List, Tuple
from .config import settings, normalize_private_key


def _to_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def _patch_clob_models():
    """Relax over-strict fields in polymarket-apis' models. The live CLOB API omits
    some fields the library marks required (e.g. blockaid_check_enabled), which would
    otherwise crash order placement with a pydantic ValidationError."""
    try:
        import polymarket_apis.types.clob_types as ct
        changed = False
        for name in ("blockaid_check_enabled",):
            f = ct.ClobMarketInfo.model_fields.get(name)
            if f is not None and f.is_required():
                f.default = False
                changed = True
        if changed:
            ct.ClobMarketInfo.model_rebuild(force=True)
    except Exception:
        pass


def _patch_market_order_rounding():
    """Cap a MARKET order's derived amount at 4 decimals.

    polymarket-apis rounds it to ROUNDING_CONFIG[tick].amount decimals, which is 5 on
    a 0.001-tick market and 6 on a 0.0001-tick one. Every Polymarket BTC 15m market is
    0.001-tick, so every live market order carried a 5-decimal share amount and the
    CLOB rejected all of them outright:

        HTTP 400 - invalid amounts, the market buy orders maker amount supports a max
        accuracy of 2 decimals, taker amount a max of 4 decimals

    4 is what the library itself uses on a 0.01-tick market, so this is its own
    rounding applied to the finer ticks — not a new rule. Market orders only; the
    limit-order path is left exactly as it was.
    """
    try:
        from dataclasses import replace
        from polymarket_apis.utilities.order_builder import builder as ob

        original = ob.OrderBuilder.get_market_order_amounts

        def capped(self, side, amount, price, round_config):
            if round_config.amount > 4:
                round_config = replace(round_config, amount=4)
            return original(self, side, amount, price, round_config)

        ob.OrderBuilder.get_market_order_amounts = capped
    except Exception:
        pass


_patch_clob_models()
_patch_market_order_rounding()


PUSD_TOKEN = "0xC011a7E12a19f7B1f670d46F03B03f3342E82DFB"
USDC_E_TOKEN = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
USDC_NATIVE_TOKEN = "0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359"

ERC20_BAL_ABI = [
    {"name": "balanceOf", "inputs": [{"name": "account", "type": "address"}], "outputs": [{"name": "", "type": "uint256"}], "type": "function"}
]


class ClobTrader:
    def __init__(self):
        self.clob = None          # PolymarketClobClient — order placement
        self.gasless = None       # PolymarketGaslessWeb3Client — relayer on-chain ops
        self.funder: Optional[str] = None         # funded wallet (deposit/proxy/safe)
        self.signature_type: Optional[int] = None
        self.ready = False
        self.last_error: Optional[str] = None
        self._approvals_done = False
        self._api_creds = None
        self._lock = threading.Lock()

    def reset(self):
        """Drop cached clients so the next call re-initialises with fresh
        credentials (call after the key / relayer settings change)."""
        with self._lock:
            self.clob = None
            self.gasless = None
            self.funder = None
            self.signature_type = None
            self.ready = False
            self.last_error = None
            self._approvals_done = False
            self._api_creds = None

    # ── wallet derivation / detection ───────────────────────────────────────────
    def _derive_creds(self):
        """CLOB API creds derived from the key (L1 auth). Cached. Doubles as the
        gasless client's builder_creds when no relayer key is set."""
        if self._api_creds is not None:
            return self._api_creds
        from polymarket_apis.clients.clob_client import PolymarketClobClient
        from eth_account import Account
        eoa = Account.from_key(settings.PRIVATE_KEY).address
        c = PolymarketClobClient(private_key=settings.PRIVATE_KEY, address=eoa,
                                 chain_id=137, signature_type=3)
        self._api_creds = c.create_or_derive_api_creds()
        return self._api_creds

    def _new_gasless(self, signature_type: int):
        return self._new_gasless_custom(signature_type, settings.PRIVATE_KEY,
                                         settings.RELAYER_API_KEY, settings.ALCHEMY_API_KEY)

    def _new_gasless_custom(self, signature_type: int, private_key: str,
                            relayer_api_key: str = "", alchemy_api_key: str = ""):
        from polymarket_apis.clients.web3_client import PolymarketGaslessWeb3Client
        from polymarket_apis.clients.clob_client import PolymarketClobClient
        from eth_account import Account

        pk = normalize_private_key(private_key)
        rk = relayer_api_key.strip() if relayer_api_key else settings.RELAYER_API_KEY
        ak = alchemy_api_key.strip() if alchemy_api_key else settings.ALCHEMY_API_KEY
        rpc = f"https://polygon-mainnet.g.alchemy.com/v2/{ak}" if ak else settings.alchemy_rpc_url()

        kwargs = {"private_key": pk, "signature_type": signature_type}
        if rk:
            kwargs["relayer_api_key"] = rk
        else:
            eoa = Account.from_key(pk).address
            c = PolymarketClobClient(private_key=pk, address=eoa, chain_id=137, signature_type=3)
            kwargs["builder_creds"] = c.create_or_derive_api_creds()
        if rpc:
            kwargs["rpc_url"] = rpc
        return PolymarketGaslessWeb3Client(**kwargs)

    def _query_address_balances(self, address: str, w3) -> Dict[str, float]:
        """Query pUSD, USDC.e, native USDC, MATIC, and Polymarket Data-API value for an address."""
        from web3 import Web3
        import httpx

        out = {
            "pusd": 0.0,
            "usdc_e": 0.0,
            "usdc_native": 0.0,
            "matic": 0.0,
            "polymarket_value": 0.0,
            "tradeable": 0.0,
        }
        if not address:
            return out
        try:
            ca = Web3.to_checksum_address(address)
        except Exception:
            return out

        try:
            pusd = w3.eth.contract(address=Web3.to_checksum_address(PUSD_TOKEN), abi=ERC20_BAL_ABI)
            out["pusd"] = float(pusd.functions.balanceOf(ca).call() / 1e6)
        except Exception:
            pass

        try:
            usdce = w3.eth.contract(address=Web3.to_checksum_address(USDC_E_TOKEN), abi=ERC20_BAL_ABI)
            out["usdc_e"] = float(usdce.functions.balanceOf(ca).call() / 1e6)
        except Exception:
            pass

        try:
            usdcnat = w3.eth.contract(address=Web3.to_checksum_address(USDC_NATIVE_TOKEN), abi=ERC20_BAL_ABI)
            out["usdc_native"] = float(usdcnat.functions.balanceOf(ca).call() / 1e6)
        except Exception:
            pass

        try:
            out["matic"] = float(w3.eth.get_balance(ca) / 1e18)
        except Exception:
            pass

        try:
            resp = httpx.get(f"https://data-api.polymarket.com/value?user={address}", timeout=3.0)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list) and data:
                    out["polymarket_value"] = float(data[0].get("value", 0.0))
        except Exception:
            pass

        out["tradeable"] = max(out["pusd"], out["usdc_e"], out["usdc_native"], out["polymarket_value"])
        return out

    def _candidate_wallets(self, gasless, eoa: Optional[str] = None) -> List[Tuple[int, str, str]]:
        """(signature_type, address, label) candidates derived from the EOA:
        - Signature Type 3: Deposit Wallet V2 (Polymarket)
        - Signature Type 0: EOA / Signer (MetaMask)
        - Signature Type 1: Poly Proxy (V1)
        - Signature Type 2: Safe Proxy
        """
        out: List[Tuple[int, str, str]] = []
        try:
            dep = gasless.get_expected_deposit_wallet()
            if dep:
                out.append((3, dep, "Deposit Wallet (V2)"))
        except Exception:
            pass

        if eoa:
            out.append((0, eoa, "EOA / Signer (MetaMask)"))

        for st, getter, label in (
            (1, gasless.get_poly_proxy_wallet_address, "Polymarket Proxy (V1)"),
            (2, gasless.get_safe_proxy_wallet_address, "Gnosis Safe"),
        ):
            try:
                addr = getter()
                if addr:
                    out.append((st, addr, label))
            except Exception:
                pass
        return out

    def _pick_funded_wallet(self, gasless, eoa: Optional[str] = None) -> Tuple[int, str]:
        """Trade from where the money actually is: pick candidate with detected funds.
        Falls back to the deposit wallet when every balance reads 0."""
        best_sig = 3
        best_funder = None
        best_bal = 0.0

        candidates = self._candidate_wallets(gasless, eoa=eoa)
        w3 = gasless.w3
        for st, addr, _ in candidates:
            bals = self._query_address_balances(addr, w3)
            bal = bals.get("tradeable", 0.0)
            if bal > best_bal:
                best_bal = bal
                best_sig = st
                best_funder = addr

        if best_funder and best_bal > 0:
            return best_sig, best_funder

        try:
            return 3, gasless.get_expected_deposit_wallet()
        except Exception:
            return 3, eoa or ""

    def _init_clients(self):
        from polymarket_apis.clients.clob_client import PolymarketClobClient
        from eth_account import Account

        eoa = Account.from_key(settings.PRIVATE_KEY).address
        probe = self._new_gasless(3)
        sig_type, funder = self._pick_funded_wallet(probe, eoa=eoa)

        gasless = probe if sig_type == 3 else self._new_gasless(sig_type)

        clob = PolymarketClobClient(
            private_key=settings.PRIVATE_KEY,
            address=funder,
            chain_id=137,
            signature_type=sig_type,
        )
        clob.set_api_creds(clob.create_or_derive_api_creds())

        self.gasless = gasless
        self.clob = clob
        self.funder = funder
        self.signature_type = sig_type
        self.ready = True
        self.last_error = None

    def ensure_ready(self) -> bool:
        if self.ready and self.clob is not None:
            return True
        with self._lock:
            if self.ready and self.clob is not None:
                return True
            if not settings.PRIVATE_KEY:
                self.last_error = "missing_private_key"
                return False
            try:
                self._init_clients()
                return True
            except Exception as e:
                self.last_error = f"{type(e).__name__}: {e}"
                self.ready = False
                self.clob = None
                return False

    def ensure_setup(self, private_key: Optional[str] = None,
                     relayer_api_key: Optional[str] = None,
                     alchemy_api_key: Optional[str] = None) -> Dict[str, Any]:
        """One-time gasless on-chain setup: deploy the deposit wallet (if needed) and
        set token approvals, sponsored by the relayer key. Required once before the
        first live order on a fresh deposit wallet."""
        pk = normalize_private_key(private_key) if private_key else settings.PRIVATE_KEY
        rk = (relayer_api_key or "").strip() if relayer_api_key is not None else settings.RELAYER_API_KEY
        ak = (alchemy_api_key or "").strip() if alchemy_api_key is not None else settings.ALCHEMY_API_KEY

        if not pk:
            return {"ok": False, "error": "missing_private_key",
                    "message": "Please enter or save your private key or seed phrase first."}
        if not rk:
            return {"ok": False, "error": "missing_relayer_api_key",
                    "message": "A Polymarket Relayer API key is required to sponsor gasless wallet deployment and token approvals."}

        try:
            gasless = self._new_gasless_custom(3, pk, rk, ak)
            receipts = gasless.set_all_approvals()
            self._approvals_done = True
            return {"ok": True, "approvals": len(receipts or [])}
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    # ── orders ──────────────────────────────────────────────────────────────────
    # A Fill-Or-Kill order either fills completely or is KILLED. The old success test
    # was `resp.get("success") is not False`, so a KILLED order was recorded as a
    # filled position — a phantom trade with no fill behind it, and `shares` was always
    # the estimate stake/quote rather than what was actually bought. Now a fill must be
    # positively proven: success is true, the status says matched/delayed, AND non-zero
    # amounts are reported on both legs. Anything else is a failure.
    _FILLED_STATUSES = ("matched", "delayed")

    def _market_order(self, token_id, amount, side: str, price) -> Dict[str, Any]:
        from polymarket_apis.types.clob_types import MarketOrderArgs, OrderType
        args = MarketOrderArgs(
            token_id=str(token_id),
            amount=round(float(amount), 2),   # BUY: USDC to spend; SELL: shares to sell
            side=side,                        # "BUY" / "SELL"
            price=round(float(price), 4) if price else 0,
            order_type=OrderType.FOK,
        )
        resp = self.clob.create_and_post_market_order(args)
        if resp is None:
            return {"ok": False, "error": "no_response_from_clob", "response": {}}

        data = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)
        order_id = (data.get("order_id") or data.get("orderID")
                    or data.get("orderId") or data.get("id"))
        status = str(data.get("status", "")).lower()
        err = data.get("error_msg") or data.get("errorMsg")

        # making/taking are the two legs of the match:
        #   BUY  -> making = USDC paid,   taking = shares received
        #   SELL -> making = shares sold, taking = USDC received
        making = _to_float(data.get("making_amount") or data.get("makingAmount"))
        taking = _to_float(data.get("taking_amount") or data.get("takingAmount"))

        if side == "BUY":
            usd, shares = making, taking
        else:
            shares, usd = making, taking

        filled = (data.get("success") is not False
                  and status in self._FILLED_STATUSES
                  and shares is not None and shares > 0
                  and usd is not None and usd > 0)

        if not filled:
            return {"ok": False, "response": data, "order_id": order_id,
                    "status": status,
                    "error": err or f"not_filled(status={status or 'unknown'})"}

        return {
            "ok": True, "response": data, "order_id": order_id, "status": status,
            # REAL fill economics — never the quote we asked for. A FOK can fill
            # anywhere up to the limit, so shares != amount/quote.
            "fill_price": (usd / shares) if shares else None,
            "fill_size": shares,
            "fill_usd": usd,
        }

    def place_market_buy(self, token_id: str, usdc_amount: float, price: Optional[float] = None) -> Dict[str, Any]:
        """Fill-Or-Kill marketable BUY for `usdc_amount` USDC of `token_id`. `price`
        is the current quote; the limit is quote + slippage buffer (capped < $1).
        On success returns the ACTUAL fill_price / fill_size / fill_usd."""
        if not token_id:
            return {"ok": False, "error": "missing_token_id"}
        if not self.ensure_ready():
            return {"ok": False, "error": self.last_error or "client_not_ready"}
        # Ensure the deposit wallet is deployed + approved before the first order.
        # A missing relayer key is tolerated: an already-set-up wallet trades fine
        # without one, and a fresh wallet will simply fail at the order instead.
        setup = self.ensure_setup()
        if not setup.get("ok") and setup.get("error") != "missing_relayer_api_key":
            return {"ok": False, "error": f"setup_failed: {setup.get('error')}"}
        try:
            limit = min(0.99, float(price) + settings.CLOB_MAX_SLIPPAGE) if price and price > 0 else 0
            return self._market_order(token_id, usdc_amount, "BUY", limit)
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    def place_market_sell(self, token_id: str, size: float, price: Optional[float] = None) -> Dict[str, Any]:
        """Fill-Or-Kill marketable SELL of `size` shares (used to exit / flip). Limit
        is quote − slippage buffer (floored at 1¢). On success returns the ACTUAL
        fill_price / fill_size / fill_usd."""
        if not token_id:
            return {"ok": False, "error": "missing_token_id"}
        if not self.ensure_ready():
            return {"ok": False, "error": self.last_error or "client_not_ready"}
        try:
            limit = max(0.01, float(price) - settings.CLOB_MAX_SLIPPAGE) if price and price > 0 else 0
            return self._market_order(token_id, size, "SELL", limit)
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    def get_last_fill(self, token_id: str) -> Optional[Dict[str, Any]]:
        """Most recent on-chain trade for this token from the CLOB's own record —
        an independent confirmation that a reported fill really happened."""
        if not self.ensure_ready():
            return None
        try:
            trades = self.clob.get_trades(token_id=str(token_id))
        except Exception:
            return None
        if not trades:
            return None
        t = trades[0]
        d = t.model_dump() if hasattr(t, "model_dump") else dict(t)
        return {"trade_id": d.get("trade_id"), "side": d.get("side"),
                "price": _to_float(d.get("price")), "size": _to_float(d.get("size")),
                "status": d.get("status"), "match_time": d.get("match_time")}

    # ── redemption ──────────────────────────────────────────────────────────────
    # A winning position does NOT become spendable money on its own: it stays a CTF
    # conditional token worth $1 that must be redeemed. Without this, live wins never
    # show up in the pUSD balance and capital quietly strands — `get_pusd_balance`
    # reads pUSD only and cannot see them.
    #
    # NOTE: redeem_position / auto_redeem_enable live on the GASLESS WEB3 client, not
    # on the CLOB client. (The build this was ported from reached for a `self.client`
    # attribute that does not exist, so both methods were dead there.)

    def enable_auto_redeem(self, private_key: Optional[str] = None) -> Dict[str, Any]:
        """Ask Polymarket to auto-redeem resolved positions. Best-effort: if the
        account doesn't support it we fall back to redeeming explicitly."""
        pk = normalize_private_key(private_key) if private_key else settings.PRIVATE_KEY
        if not pk:
            return {"ok": False, "error": "missing_private_key",
                    "message": "Please enter or save your private key or seed phrase first."}
        try:
            gasless = self.gasless if (self.ready and self.gasless is not None and not private_key) else self._new_gasless_custom(3, pk)
            if not hasattr(gasless, "auto_redeem_enable"):
                return {"ok": False, "error": "auto_redeem_unsupported"}
            res = gasless.auto_redeem_enable()
            return {"ok": True, "result": str(res)}
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    def redeem(self, condition_id: str, amounts, neg_risk: bool = False) -> Dict[str, Any]:
        """Redeem a resolved position into pUSD.

        `amounts` is the per-outcome share list the CTF expects ([up, down]); the
        losing leg is simply zero.
        """
        if not condition_id:
            return {"ok": False, "error": "missing_condition_id"}
        if not self.ensure_ready():
            return {"ok": False, "error": self.last_error or "client_not_ready"}
        if not hasattr(self.gasless, "redeem_position"):
            return {"ok": False, "error": "redeem_unsupported_by_client"}
        try:
            amts = [float(a) for a in (amounts if isinstance(amounts, (list, tuple)) else [amounts])]
            tx = self.gasless.redeem_position(condition_id, amts, neg_risk)
            return {"ok": True, "tx": str(tx)}
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    # ── withdrawal (auto capital extractor) ─────────────────────────────────────
    def withdraw_pusd(self, recipient: str, amount: float) -> Dict[str, Any]:
        """Transfer `amount` pUSD from the funded (deposit) wallet to `recipient`.
        Gasless via the relayer. Used by the auto-withdrawal state machine."""
        if not recipient:
            return {"ok": False, "error": "missing_withdraw_address"}
        if amount is None or float(amount) <= 0:
            return {"ok": False, "error": "invalid_amount"}
        if not self.ensure_ready():
            return {"ok": False, "error": self.last_error or "client_not_ready"}
        if not hasattr(self.gasless, "transfer_pusd"):
            return {"ok": False, "error": "withdraw_unsupported_by_client"}
        try:
            try:
                from eth_utils import to_checksum_address
                addr = to_checksum_address(recipient)
            except Exception:
                addr = recipient
            receipt = self.gasless.transfer_pusd(addr, round(float(amount), 2))
            tx = None
            try:
                data = receipt.model_dump() if hasattr(receipt, "model_dump") else dict(receipt)
                tx = data.get("transaction_hash") or data.get("transactionHash") or data.get("hash")
            except Exception:
                tx = str(receipt) if receipt is not None else None
            return {"ok": True, "tx": tx, "amount": round(float(amount), 2), "recipient": addr}
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    def is_tx_confirmed(self, tx_hash: str) -> Optional[bool]:
        """True/False once the receipt is readable, None while still unknown. Used by
        the capital extractor when `resume_after` is set to `confirmed`."""
        if not tx_hash:
            return None
        try:
            from web3 import Web3
            rpc = settings.alchemy_rpc_url() or settings.POLYGON_RPC_URL
            if not rpc:
                return None
            w3 = Web3(Web3.HTTPProvider(rpc, request_kwargs={"timeout": 6.0}))
            receipt = w3.eth.get_transaction_receipt(tx_hash)
            if receipt is None:
                return None
            return bool(receipt.get("status", 0) == 1)
        except Exception:
            return None

    # ── diagnostics / balance ───────────────────────────────────────────────────
    def get_eoa_address(self) -> Optional[str]:
        """The EOA address derived from PRIVATE_KEY — the wallet you control (the
        key/seed owner). It signs orders but holds no funds or gas."""
        if not settings.PRIVATE_KEY:
            return None
        try:
            from eth_account import Account
            return Account.from_key(settings.PRIVATE_KEY).address
        except Exception:
            return None

    def test_connection(self, private_key: Optional[str] = None,
                        relayer_api_key: Optional[str] = None,
                        alchemy_api_key: Optional[str] = None) -> Dict[str, Any]:
        """Derive the EOA + candidate wallets (Deposit V2, EOA, Proxy, Safe) and report
        balances across pUSD, USDC.e, native USDC, and Polymarket Data-API value.
        Read-only (no relayer key required). Supports testing unsaved credentials."""
        from concurrent.futures import ThreadPoolExecutor

        pk = normalize_private_key(private_key) if private_key else settings.PRIVATE_KEY
        if not pk:
            return {"ok": False, "error": "missing_private_key",
                    "message": "Please enter a private key or seed phrase."}
        try:
            from eth_account import Account
            eoa = Account.from_key(pk).address
        except Exception as e:
            return {"ok": False, "error": f"invalid_key: {type(e).__name__}: {e}"}
        try:
            rk = (relayer_api_key or "").strip() if relayer_api_key is not None else settings.RELAYER_API_KEY
            ak = (alchemy_api_key or "").strip() if alchemy_api_key is not None else settings.ALCHEMY_API_KEY
            probe = self._new_gasless_custom(3, pk, rk, ak)

            candidates = self._candidate_wallets(probe, eoa=eoa)
            w3 = probe.w3

            def fetch_wallet_info(item):
                st, addr, label = item
                bals = self._query_address_balances(addr, w3)
                return {
                    "signature_type": st,
                    "address": addr,
                    "label": label,
                    "pusd_balance": bals["pusd"],
                    "usdce_balance": bals["usdc_e"],
                    "usdc_native_balance": bals["usdc_native"],
                    "polymarket_value": bals["polymarket_value"],
                    "matic_balance": bals["matic"],
                    "tradeable_balance": bals["tradeable"],
                    "has_funds": bals["tradeable"] > 0
                }

            with ThreadPoolExecutor(max_workers=min(4, max(1, len(candidates)))) as pool:
                wallets = list(pool.map(fetch_wallet_info, candidates))

            best_sig = 3
            best_funder = None
            best_bal = 0.0
            total_detected = 0.0

            for w in wallets:
                tb = w["tradeable_balance"]
                total_detected += tb
                if tb > best_bal:
                    best_bal = tb
                    best_sig = w["signature_type"]
                    best_funder = w["address"]

            if best_funder is None or best_bal <= 0:
                best_sig = 3
                try:
                    best_funder = probe.get_expected_deposit_wallet()
                except Exception:
                    best_funder = eoa

            advisory = ""
            if total_detected > 0:
                if best_sig == 3:
                    advisory = f"Funds detected in your Polymarket Deposit Wallet (${best_bal:.2f}). Ready for live trading!"
                elif best_sig == 0:
                    advisory = f"Funds detected in your EOA (${best_bal:.2f}). To trade gasless on Polymarket V2, deposit your USDC to your Polymarket Deposit Wallet address ({best_funder})."
                else:
                    advisory = f"Funds detected in your Proxy/Safe wallet (${best_bal:.2f})."
            else:
                advisory = "No funds detected across your derived Polygon addresses ($0.00). If you see funds on polymarket.com, did you sign in with Google or Email? (Google logins use an embedded Magic wallet with a different address). Check your address on polymarket.com -> Profile."

            return {
                "ok": True,
                "eoa": eoa,
                "chosen_signature_type": best_sig,
                "funder": best_funder,
                "relayer_key_set": bool(rk),
                "total_detected_funds": round(total_detected, 2),
                "advisory": advisory,
                "wallets": wallets,
            }
        except Exception as e:
            return {"ok": False, "eoa": eoa, "error": f"{type(e).__name__}: {e}"}

    def get_usdc_balance(self) -> Optional[float]:
        """Tradeable balance of the funded wallet (dollars), checking pUSD, USDC.e,
        native USDC, and Polymarket portfolio value."""
        if not settings.PRIVATE_KEY:
            return None
        try:
            from eth_account import Account
            eoa = Account.from_key(settings.PRIVATE_KEY).address
            if self.ready and self.gasless is not None and self.funder:
                bals = self._query_address_balances(self.funder, self.gasless.w3)
                return float(bals.get("tradeable", 0.0))
            probe = self._new_gasless(3)
            sig_type, funder = self._pick_funded_wallet(probe, eoa=eoa)
            bals = self._query_address_balances(funder, probe.w3)
            return float(bals.get("tradeable", 0.0))
        except Exception:
            return None


clob_trader = ClobTrader()
