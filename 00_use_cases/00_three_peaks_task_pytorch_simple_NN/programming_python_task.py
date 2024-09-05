"""
Given an array nums of size n and consisting of natural numbers from 1 to n, return the duplicate number.

It is known all the numbers in nums are unique except for one duplicate.

test cases:

Input: nums = [3, 1, 3, 5, 2]. Output: 3
Input: nums = [2, 3, 4, 2]. Output: 2
Input: nums = [8, 5, 7, 6, 3, 4, 5, 2, 9, 1]. Output: 5
Input: nums = [5, 1, 4, 7, 3, 10, 8, 9, 10, 2]. Output: 10
"""

def my_function(test_list: list):
    
    set_nums = set(test_list)

    for i in test_list:
        if i in set_nums:
            return i

"""The optimal solution is the following:
- no additional mem alloc
- no set
- use the current index: check if the number @ current index is negative.
If it is negative, it has already been "marked" as "used". Hence it is the duplicate.
If it is positive, make it negative in order to mark it. Then continue to the next index.
"""

def optimal_solution(test_list: list):
    for idx in range(0, len(test_list)):
        if test_list[test_list[idx]-1] < 0:
            print("Exit route, idx: ", idx)
            print("value: ", test_list[idx])
            return (-test_list[idx])
        else:
            print("index: ", idx)
            print("value at index: ", test_list[idx])
            print("test-list:", test_list)
            test_list[test_list[idx]-1] = test_list[test_list[idx]-1] * -1
            print("TL after change: ", test_list)


if __name__ == "__main__":
    nums = [2, 3, 4, 2]
    check_number = my_function(nums)
    optimal_number = optimal_solution(nums)

    print(check_number)
    print(optimal_number)