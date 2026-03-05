from dataclasses import dataclass
from xml.etree.ElementTree import Element


@dataclass
class XODRPolynomial:
    """
    Multiple OpenDRIVE elements use polynomial coefficients, e.g. Elevation, LaneOffset, etc.
    This class provides a common interface to parse and access polynomial coefficients.

    Stores a  polynomial function of the third order:
    value(ds) = a + b*ds + c*ds² + d*ds³
    """

    s: float
    a: float
    b: float
    c: float
    d: float

    @classmethod
    def parse(cls: type["XODRPolynomial"], element: Element) -> "XODRPolynomial":
        args = {key: float(element.get(key)) for key in ["s", "a", "b", "c", "d"]}
        return cls(**args)

    def get_value(self, ds: float) -> float:
        """
        Evaluate the polynomial at a given ds value.
        :param ds: The distance along the road.
        :return: The evaluated polynomial value.
        """
        return self.a + self.b * ds + self.c * ds**2 + self.d * ds**3
