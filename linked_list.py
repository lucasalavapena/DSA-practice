# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

def linked_list_as_cycle(head: ListNode| None) -> bool:
    
    slow = head
    fast = head

    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

        if slow == fast:
            return True

    return False

def reverse_linked_list(head: ListNode| None) -> ListNode | None:
    curr = None
    nxt = head

    while nxt:
        temp = nxt.next
        nxt.next = curr

        curr = nxt
        nxt = temp
    
    return curr