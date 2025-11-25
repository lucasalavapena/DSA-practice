

def cycle_sort(nums: list[int]) -> None:
    n = len(nums)
    
    for i in range(n):
        while nums[i]-1 in range(n) and nums[i] != nums[nums[i]-1]:
            nums[nums[i]-1], nums[i] = nums[i], nums[nums[i]-1]