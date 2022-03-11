from classes.dice_set import DiceSet

class KniffelOptionClass:
    name: str
    is_possible = False
    points = 0
    dice_set = None
    
    def __init__(self, name: str, points: int, ds: DiceSet, is_possible: bool = False):
        self.name = name
        self.is_possible = True if points > 0 else False
        self.points = points
        self.dice_set = ds
        
    def __repr__(self):
        return "{name: '" + str(self.name) + "', is_possible: '" + str(self.is_possible) + "', points: '" + str(self.points) + "'}"
    def __str__(self):
        return "Selected '" + str(self.name) + "' and got " + str(self.points) + " points."