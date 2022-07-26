from src.kniffel.classes.dice_set import DiceSet
from src.kniffel.classes.options import KniffelOptions


class KniffelOptionClass:
    name: str
    is_possible = False
    points = 0
    dice_set = None
    id = -1

    def __init__(
        self,
        name: str,
        points: int,
        ds: DiceSet,
        id: KniffelOptions,
        is_possible: bool = False,
    ):
        self.name = name

        possible = True if points > 0 else False

        if is_possible == False:
            self.is_possible = possible
        else:
            self.is_possible = is_possible

        self.points = points
        self.dice_set = ds
        self.id = id.value

    def get_name(self):
        return self.name

    def get_id(self):
        return self.id

    def get_points(self):
        return self.points

    def __repr__(self):
        return (
            "{name: '"
            + str(self.name)
            + "', is_possible: '"
            + str(self.is_possible)
            + "', points: '"
            + str(self.points)
            + "', id: '"
            + str(self.id)
            + "'}"
        )

    def __str__(self):
        return (
            "Selected '" + str(self.name) + "' and got " + str(self.points) + " points."
        )
