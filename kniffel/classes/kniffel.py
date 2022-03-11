from classes.status import KniffelStatus
from classes.options import KniffelOptions
from classes.dice_set import DiceSet
from classes.attempt import Attempt
from classes.kniffel_check import KniffelCheck
from classes.kniffel_option import KniffelOptionClass

class Kniffel:
    turns = []
    logging = False
    
    def __init__(self, logging: bool = False):
        self.turns = []
        self.logging = logging
    
    def add_turn(self, keep: list = None):
        """
        Add turn
        
        :param list keep: hot encoded list of which dice to keep (1 = keep, 0 = drop)
        """
        if self.turns_left > 0:
            if self.is_new_game() == True or self.is_turn_finished() == True:
                self.turns.append(Attempt())
        
            self.turns[-1].add_attempt(keep)
        else:
            raise Exception('Cannot play more then 13 rounds. Play a new game!')
          
    def finish_turn(self, option: KniffelOptions):
        """
        Finish turn
        
        :param KniffelOptions option: selected option how to finish the turn
        """
        if self.is_option_possible(option) is True:
            if self.is_new_game() == False and self.is_turn_finished() == False:
                self.turns[-1].finish_attempt(option)
        else:
            raise Exception('Cannot select the same Option again. Select another Option!')
    
    def get_points(self):
        """
        Get the total points
        """
        total = 0
        for turn in self.turns:
            if turn.status is KniffelStatus.FINISHED:
                total += turn.selected_option.points
        
        if self.is_bonus() is True:
            total += 35
        
        return total
    
    def is_option_possible(self, option: KniffelOptions):
        """
        Is Option possible
        
        :param KniffelOptions option: kniffel option to check
        """
        for turn in self.turns:
            if turn.option is option:
                return False
            
        return True
    
    def is_bonus(self):
        """
        Is bonus possible. 
        Sum for einser, zweier, dreier, vierer, fünfer und secher needs to be higher or equal to 63
        """
        total = 0
        for turn in self.turns:
            if turn.status is KniffelStatus.FINISHED and (turn.option is KniffelOptions.ONES or turn.option is KniffelOptions.TWOS or turn.option is KniffelOptions.THREES or turn.option is KniffelOptions.FOURS or turn.option is KniffelOptions.FIVES or turn.option is KniffelOptions.SIXES):
                total += turn.selected_option.points
        
        return True if total >= 63 else False
    
    def turns_left(self):
        """
        How many turns are left
        """
        return 13 - len(self.turns)
    
    def is_new_game(self):
        """
        Is the game new
        """
        if len(self.turns) == 0:
            return True
        else:
            return False
        
    def is_turn_finished(self):
        """
        Is the turn finished
        """
        if self.is_new_game() == False:
            if self.turns[-1].status == KniffelStatus.FINISHED:
                return True
            else:
                return False
        else:
            return True
    
    def check(self):
        """
        Check latest dice set for possible points
        """        
        
        ds = self.turns[-1][-1]
        values = ds.to_list()
        
        check = dict()
        
        check["ones"] = KniffelCheck().check_1(ds)
        check["twos"] = KniffelCheck().check_2(ds)
        check["threes"] = KniffelCheck().check_3(ds)
        check["fours"] = KniffelCheck().check_4(ds)
        check["fives"] = KniffelCheck().check_5(ds)
        check["sixes"] = KniffelCheck().check_6(ds)
        check["three-time"] = KniffelCheck().check_three_times(ds)
        check["four-time"] = KniffelCheck().check_four_times(ds)
        check["full-house"] = KniffelCheck().check_full_house(ds)
        check["small-street"] = KniffelCheck().check_small_street(ds)
        check["large-street"] = KniffelCheck().check_large_street(ds)
        check["kniffel"] = KniffelCheck().check_kniffel(ds)
        check["chance"] = KniffelCheck().check_chance(ds)
        
        return check 
    
    def mock(self, mock: DiceSet):
        """
        Mock turn play
        
        :param DiceSet mock: mock dice set
        """
        if self.is_new_game() is True or self.is_turn_finished() is True:
            self.turns.append(Attempt())
          
        self.turns[-1].mock(mock)
        
    def print_check(self):
        """
        Print the check of possible options
        """
        options = { k: v for k, v in self.check().items() if v.is_possible == True }
        print(options)
        
    def to_list(self):
        """
        Transform list of dice objects to simple int array list
        """
        values = []
        for v in self.turns:
            values.append(v.to_list())            
        return values    
    
    def print(self):
        """
        Print list
        """
        i = 1
        for round in self.turns:
            print("# Turn: " + str(i) + "/13")
            round.print()
            i += 1