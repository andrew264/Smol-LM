from torch import Tensor
from transformers import LogitsWarper


class TemperatureRangeLogitsWarper(LogitsWarper):
    """
    A logits warper that adjusts the temperature of the logits over a specified number of steps.
    """

    def __init__(self, start: float, end: float, num_steps: int):
        """
        Initializes the TemperatureRangeLogitsWarper with the specified start and end temperatures and the number of steps.

        Args:
            start (float): The starting temperature.
            end (float): The ending temperature.
            num_steps (int): The number of steps over which to change the temperature.

        Raises:
            ValueError: If either the start or end temperature is less than 0.
        """
        if end < 0 or start < 0:
            raise ValueError("Temperature must be greater than 0.")
        self.start = start
        self.end = end
        self.num_steps = num_steps
        self._step = (end - start) / num_steps
        self._current_step = 0

    def _get_temperature(self) -> float:
        if self._current_step >= self.num_steps:
            return self.end
        temp = self.start + self._current_step * self._step
        self._current_step += 1
        return temp

    def __call__(self, input_ids: Tensor, scores: Tensor) -> Tensor:
        temperature = self._get_temperature()
        scores_processed = scores / temperature
        return scores_processed
