from heapq import heappop, heappush


def dijkstra_core(n: int, adj: dict[int, set], source: int, dest: int) -> list[int]:
    dist = [float("inf")] * n # or have an empty set for a adj with no edges
    dist[source] = 0
    heap = [(0, source)] 

    while heap:
        curr_dist, node = heappop(heap)

        if node == dest:
            return curr_dist
        for (neigh, neigh_w) in adj[node]:
            cand_dist = curr_dist + neigh_w
            if cand_dist < dist[neigh]:
                dist[neigh] = cand_dist
                heappush(heap, (cand_dist, neigh))
    return dist

