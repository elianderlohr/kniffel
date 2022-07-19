class GameFinishedException(Exception):
    def __init__(self, message="Game finished!"):
        super(GameFinishedException, self).__init__(message)


class NewGameException(Exception):
    def __init__(self, message="Cannot finish new game!"):
        super(NewGameException, self).__init__(message)


class TurnFinishedException(Exception):
    def __init__(self, message="Cannot do more then 3 attempts per round."):
        super(TurnFinishedException, self).__init__(message)


class SelectedOptionException(Exception):
    def __init__(
        self,
        message="Cannot select the same Option again or not possible for this. Select another Option!",
    ):
        super(SelectedOptionException, self).__init__(message)
