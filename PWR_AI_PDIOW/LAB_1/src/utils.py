from dataclasses import dataclass


@dataclass
class Container:
    data1: str
    data2: int


class TaskExecuter:
    def check_if_positive(self, number):
        if number > 0:
            print(f"Number {number} is positive")
        else:
            print(f"Number {number} is negative")

    def print_range(self):
        for each in range(5):
            print(f"Element {each+1}")

    def return_doubled(self, number):
        return number * 2

    def create_tuple_from_list(self, lists):
        return tuple(lists)

    def is_elem_from_list_in_dict(self, mydict, mylist, key):
        for elem in mylist:
            if mydict[key] == elem:
                return True
        return False
