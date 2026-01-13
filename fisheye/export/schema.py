from typing import NamedTuple, Any


class FCColumn(NamedTuple):
    width: int
    default: Any


FC_SCHEMA = {
    "File": FCColumn(4, 1),
    "Total": FCColumn(7, 0),
    "Frame#": FCColumn(8, 0),
    "Dir": FCColumn(5, ""),
    "R (m)": FCColumn(8, 0.0),
    "Theta": FCColumn(8, 0.0),
    "L(cm)": FCColumn(8, 0.0),
    "dR(cm)": FCColumn(8, 0.0),
    "L/dR": FCColumn(8, 0.0),
    "Aspect": FCColumn(8, 0.0),
    "Time": FCColumn(10, "00:00:00"),
    "Date": FCColumn(12, ""),
    "Latitude": FCColumn(19, "N 00 d  0.00000 m"),
    "Longitude": FCColumn(20, "E 000 d  0.00000 m"),
    "Pan": FCColumn(9, 0.0),
    "Tilt": FCColumn(9, 0.0),
    "Roll": FCColumn(9, 0.0),
    "Species": FCColumn(10, "Unknown"),
    "Motion": FCColumn(39, "Running <-->"),
    "Q": FCColumn(7, 5),
    "N": FCColumn(4, 1),
    "Comment": FCColumn(10, ""),
}
