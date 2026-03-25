import random

def cycle_sort(nums: list[int]) -> None:
    n = len(nums)
    
    for i in range(n):
        while nums[i]-1 in range(n) and nums[i] != nums[nums[i]-1]:
            nums[nums[i]-1], nums[i] = nums[i], nums[nums[i]-1]


def sortColors(nums: list[int]) -> None:
    """
    dutch flag problem
    Do not return anything, modify nums in-place instead.
    """
    red, white, blue = 0, 0, len(nums) -1
    
    while white <= blue:
        if nums[white] == 0:
            nums[white], nums[red] = nums[red], nums[white]
            white += 1
            red += 1
        elif nums[white] == 1:
            white += 1
        else:
            nums[white], nums[blue] = nums[blue], nums[white]
            blue -= 1

def findKthLargest(nums: list[int], k: int) -> int:
    if not nums:
        return 

    pivot = random.choice(nums)
    left = [n for n in nums if n > pivot]
    mid =  [n for n in nums if n == pivot]
    right = [n for n in nums if n < pivot]
    L, M = len(left), len(mid)
    
    if k <= L:
        return self.findKthLargest(left, k)
    elif k > L + M:
        return self.findKthLargest(right, k-L-M)
    else:
        return pivot



class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        return self.quicksort(nums)
        
    def mergeSort(self, nums: List[int]) -> List[int]:
        l = len(nums)
        if l > 1:
            m = l // 2
            L = nums[:m]
            R = nums[m:]
            self.mergeSort(L)
            self.mergeSort(R)
            
            # merge two sorted list
            i = j = k = 0
            while i < len(L) and j < len(R):
                if L[i] < R[j]:
                    nums[k] = L[i]
                    i += 1
                else:
                    nums[k] = R[j]
                    j += 1
                k += 1
            nums[k:] = L[i:] + R[j:]
            # print(m, nums)
            
    def quicksort(self, arr):
        n = len(arr)
        
        if n < 2:
            return arr
        
        pivot_loc = random.randint(0, n-1)
        
        arr[0], arr[pivot_loc] = arr[pivot_loc], arr[0] 
        pivot_loc = 0
        
        for i in range(1, n):
            if arr[i] <= arr[0] :
                pivot_loc += 1
                arr[i], arr[pivot_loc] = arr[pivot_loc], arr[i] 
                
        arr[0], arr[pivot_loc] = arr[pivot_loc], arr[0] 
                
            
        left = self.quicksort(arr[0:pivot_loc])
        right = self.quicksort(arr[pivot_loc+1:])

        return left + [arr[pivot_loc]] + right
        
        
