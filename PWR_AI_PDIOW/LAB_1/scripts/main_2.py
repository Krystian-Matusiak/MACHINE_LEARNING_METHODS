import src.utils

te = src.utils.TaskExecuter()

te.check_if_positive(10)
te.check_if_positive(-10)

te.print_range()

print(te.return_doubled(4))

tup = te.create_tuple_from_list([1, 2, 3])
print(tup)
print(type(tup))

l = [1, 2, 3, 4, 5, 6, 7]
l2 = [1, 2, 3, 4, 5, 6]
d = {"first": 11, "second": 7, "third": 33}
key = "second"
print(te.is_elem_from_list_in_dict(d, l, key))
print(te.is_elem_from_list_in_dict(d, l2, key))

c = src.utils.Container("test_container", 666)
print(c)
