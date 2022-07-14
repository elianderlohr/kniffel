import random

class Dice:
    value = 0
    logging = False
    
    def __init__(self, mock: int = -1, logging: bool = False):
        random.seed()
        
        if (mock > 0):
            self.value = mock
        else:
            self.roll()
            
        self.logging = logging
        
    def roll(self):
        """
        Roll dice and set random val between 1 and 6
        """
        self.value = random.randint(1,6)
        
    def get(self):
        """
        Get value of the dice as int
        """
        return self.value