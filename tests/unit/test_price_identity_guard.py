"""A price ratio catches ticker collisions that a symbol check cannot.

The identity guards in the expansion pipeline are: CoinGecko must list exactly one token
with that ticker on the chain, and the contract's own symbol() must match the exchange's
ticker. Both passed for MET, and MET is a different asset:

    MET/WETH on Ethereum, pool 0xCEb5c29bdE4604296135DD7b027A433fD3633516
      pool     0.0007468546 WETH per MET
      Binance  0.0000848547 WETH per MET
      ratio    8.80x        -> reported as +78,008 bps of dislocation

The token at 0x2Ebd53d0...89aa really is called MET on chain, and Binance really does
list a MET. They are not the same project. No amount of symbol comparison can tell them
apart, because both symbols are correct.

What does tell them apart is the price. Two venues quoting the SAME asset agree to within
a few percent, always -- that is what arbitrage means, and the entire premise of this
strategy is that they agree to within basis points. So a ratio of 8.8 is not a
dislocation to be ranked; it is proof the two sides are quoting different things.

This is the last guard in the sequence, and the only one that works on the failure the
others cannot see. It is also the one that would have put the largest apparent
opportunity in the dataset in the bin, which is exactly why it has to be automatic rather
than left to whoever reads the table.

Deliberately loose. A 2x band would reject nothing real: a genuine dislocation of 100%
between a CEX and a DEX quoting the same liquid asset does not happen, and if it did the
correct response would still be to disbelieve the data first.
"""
from decimal import Decimal

import pytest

from src.research.observations import plausible_same_asset


class TestItAcceptsRealMarkets:
    @pytest.mark.parametrize("ratio", ["1.0", "1.0005", "0.9995", "1.05", "0.95"])
    def test_prices_that_agree_are_plausible(self, ratio):
        assert plausible_same_asset(
            Decimal("100") * Decimal(ratio), Decimal("100")
        ) is True

    def test_the_measured_bnb_gap_is_plausible_as_a_price(self):
        """BNB showed a 1.0456x ratio -- 456 bps. Implausible as an arbitrage and
        entirely plausible as a price, which is a different judgement and not this
        function's job. It rejects only what cannot be the same asset."""
        assert plausible_same_asset(
            Decimal("0.3326363661"), Decimal("0.3181367795")
        ) is True

    def test_a_large_but_believable_dislocation_is_still_plausible(self):
        """20% is enormous for this strategy and still a price rather than evidence of a
        different asset. Rejecting it would hide a real finding."""
        assert plausible_same_asset(Decimal("120"), Decimal("100")) is True


class TestItRejectsCollisions:
    def test_the_measured_met_collision_is_rejected(self):
        assert plausible_same_asset(
            Decimal("0.0007468546"), Decimal("0.0000848547")
        ) is False

    @pytest.mark.parametrize("factor", ["3", "10", "100", "1000"])
    def test_a_large_multiple_is_rejected(self, factor):
        assert plausible_same_asset(
            Decimal("100") * Decimal(factor), Decimal("100")
        ) is False

    def test_the_rejection_is_symmetric(self):
        """A collision found from the other side is the same collision. An asymmetric
        band would catch it only when the pool happened to be the richer venue."""
        assert plausible_same_asset(Decimal("100"), Decimal("880")) is False
        assert plausible_same_asset(Decimal("880"), Decimal("100")) is False


class TestHonestEdges:
    def test_a_zero_price_is_not_plausible(self):
        assert plausible_same_asset(Decimal("0"), Decimal("100")) is False
        assert plausible_same_asset(Decimal("100"), Decimal("0")) is False

    def test_a_negative_price_is_not_plausible(self):
        assert plausible_same_asset(Decimal("-100"), Decimal("100")) is False

    def test_none_is_not_plausible(self):
        assert plausible_same_asset(None, Decimal("100")) is False
        assert plausible_same_asset(Decimal("100"), None) is False

    def test_the_band_is_configurable_and_documented(self):
        """Tightening it is a judgement about what a real market can do, so it has to be
        adjustable -- and the default has to be stated where it is used."""
        assert plausible_same_asset(
            Decimal("150"), Decimal("100"), max_ratio=Decimal("1.2")
        ) is False
        assert plausible_same_asset(
            Decimal("150"), Decimal("100"), max_ratio=Decimal("2")
        ) is True

    def test_a_ratio_band_below_one_is_rejected_as_a_setting(self):
        with pytest.raises(ValueError):
            plausible_same_asset(
                Decimal("100"), Decimal("100"), max_ratio=Decimal("0.5")
            )
