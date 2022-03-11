from classes.status import KniffelStatus
from classes.options import KniffelOptions
from classes.dice_set import DiceSet
from classes.kniffel_option import KniffelOptionClass

class KniffelCheck():
    
    def occures_n_times(self, ds: DiceSet, n: int, blacklist: list = []):
        points = 0
        
        base_list = [1,2,3,4,5,6]
        
        c = [x for x in base_list if x not in blacklist]
        
        dice_list = ds.to_list()
        for v in c:
            if dice_list.count(v) >= n:
                return True
    
    def what_occures_n_times(self, ds: DiceSet, n: int):
        dice_list = ds.to_list()
        if dice_list.count(1) >= n:
            return 1
        if dice_list.count(2) >= n:
            return 2
        if dice_list.count(3) >= n:
            return 3
        if dice_list.count(4) >= n:
            return 4
        if dice_list.count(5) >= n:
            return 5
        if dice_list.count(6) >= n:
            return 6
        
        return -1
    
    def check_1(self, ds: DiceSet):
        """
        Check the inserted dice set for "Einser"
        
        :param DiceSet ds: set of dices
        """
        c = ds.to_list().count(1)
        return KniffelOptionClass("ones", c, ds=ds)
    
    def check_2(self, ds: DiceSet):
        """
        Check the inserted dice set for "Zweier"
        
        :param DiceSet ds: set of dices
        """
        c = ds.to_list().count(2)
        return KniffelOptionClass("twos", c * 2, ds=ds)
    
    def check_3(self, ds: DiceSet):
        """
        Check the inserted dice set for "Dreier"
        
        :param DiceSet ds: set of dices
        """
        c = ds.to_list().count(3)
        return KniffelOptionClass("threes", c * 3, ds=ds)
    
    def check_4(self, ds: DiceSet):
        """
        Check the inserted dice set for "Vierer"
        
        :param DiceSet ds: set of dices
        """
        c = ds.to_list().count(4)
        return KniffelOptionClass("fours", c * 4, ds=ds)
    
    def check_5(self, ds: DiceSet):
        """
        Check the inserted dice set for "Fünfer"
        
        :param DiceSet ds: set of dices
        """
        c = ds.to_list().count(5)
        return KniffelOptionClass("fives", c * 5, ds=ds)
    
    def check_6(self, ds: DiceSet):
        """
        Check the inserted dice set for "Sechser"
        
        :param DiceSet ds: set of dices
        """
        c = ds.to_list().count(6)
        return KniffelOptionClass("sixes", c * 6, ds=ds)  
    
    def check_three_times(self, ds: DiceSet):
        """
        Check the inserted dice set for a "Dreierpasch"
        
        :param DiceSet ds: set of dices
        """
        has_three_same = True if self.occures_n_times(ds, 3) else False
        
        if has_three_same == True:
            return KniffelOptionClass(name = "three-times", points = sum(ds.to_list()), is_possible = True, ds=ds) 
        else:
            return KniffelOptionClass(name = "three-times", points = 0, is_possible = False, ds=ds) 
    
    def check_four_times(self, ds: DiceSet):
        """
        Check the inserted dice set for a "Viererpasch"
        
        :param DiceSet ds: set of dices
        """
        has_four_same = True if self.occures_n_times(ds, 4) else False
        
        if has_four_same == True:
            return KniffelOptionClass(name = "four-times", points = sum(ds.to_list()), is_possible = True, ds=ds) 
        else:
            return KniffelOptionClass(name = "four-times", points = 0, is_possible = False, ds=ds)
    
    def check_full_house(self, ds: DiceSet):
        """
        Check the inserted dice set for a "Full House"
        
        :param DiceSet ds: set of dices
        """
        three_times = self.what_occures_n_times(ds, 3)

        has_three_same = True if self.occures_n_times(ds, 3) else False
        has_two_same = True if self.occures_n_times(ds, 2, blacklist=[three_times]) else False
        
        if has_three_same == True and has_two_same == True:
            return KniffelOptionClass(name = "full-house", points = 25, is_possible = True, ds=ds) 
        else:
            return KniffelOptionClass(name = "full-house", points = 0, is_possible = False, ds=ds) 
    
    def check_small_street(self, ds: DiceSet):
        """
        Check the inserted dice set for a "Kleine Straße"
        
        :param DiceSet ds: set of dices
        """
        dice_list = ds.to_list()
    
        has_small_street = False
        
        if (
            (dice_list.count(1) >= 1 and dice_list.count(2) >= 1 and dice_list.count(3) >= 1 and dice_list.count(4) >= 1) or
            (dice_list.count(2) >= 1 and dice_list.count(3) >= 1 and dice_list.count(4) >= 1 and dice_list.count(5) >= 1) or
            ( dice_list.count(3) >= 1 and dice_list.count(4) >= 1 and dice_list.count(5) >= 1 and dice_list.count(6) >= 1)
            ):
            has_small_street = True
            
        if has_small_street == True:
            return KniffelOptionClass(name = "small-street", points = 30, is_possible = True, ds=ds) 
        else:
            return KniffelOptionClass(name = "small-street", points = 0, is_possible = False, ds=ds)
            
    def check_large_street(self, ds: DiceSet):
        """
        Check the inserted dice set for a "Große Straße"
        
        :param DiceSet ds: set of dices
        """
        dice_list = ds.to_list()
    
        has_large_street = False
        
        if (
            (dice_list.count(1) >= 1 and dice_list.count(2) >= 1 and dice_list.count(3) >= 1 and dice_list.count(4) >= 1 and dice_list.count(5) >= 1) or
            (dice_list.count(2) >= 1 and dice_list.count(3) >= 1 and dice_list.count(4) >= 1 and dice_list.count(5) >= 1 and dice_list.count(6) >= 1)
            ):
            has_large_street = True
            
        if has_large_street == True:
            return KniffelOptionClass(name = "large-street", points = 40, is_possible = True, ds=ds) 
        else:
            return KniffelOptionClass(name = "large-street", points = 0, is_possible = False, ds=ds)
            
    
    def check_kniffel(self, ds: DiceSet):
        """
        Check the inserted dice set for a "Kniffel"
        
        :param DiceSet ds: set of dices
        """
        has_kniffel = True if self.occures_n_times(ds, 5) else False
        
        if has_kniffel == True:
            return KniffelOptionClass(name = "kniffel", points = 50, is_possible = True, ds=ds) 
        else:
            return KniffelOptionClass(name = "kniffel", points = 0, is_possible = False, ds=ds)
        
    def check_chance(self, ds: DiceSet):
        """
        Check the inserted dice set for a "Chance"
        
        :param DiceSet ds: set of dices
        """
        return KniffelOptionClass(name = "chance", points = sum(ds.to_list()), is_possible = True, ds=ds)