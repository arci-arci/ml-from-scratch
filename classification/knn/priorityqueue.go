package knn

import "math"

type PriorityItem struct {
	// priority is equal to the distance
	// between a target point and a point inside
	// the ball tree
	Priority float64
	Value    NormalizedDocument
	Position int
}

type PriorityQueue struct {
	Capacity int
	Len      int
	Items    []PriorityItem
}

func CreatePriorityQueue(size int) PriorityQueue {
	items := make([]PriorityItem, size+1)

	return PriorityQueue{
		Capacity: size + 1,
		Len:      0,
		Items:    items,
	}
}

func (Q *PriorityQueue) IsEmpty() bool {
	return Q.Size() == 0
}

func (Q *PriorityQueue) Size() int {
	return Q.Len
}

func (Q *PriorityQueue) Min() NormalizedDocument {
	return Q.Items[1].Value
}

func (Q *PriorityQueue) Insert(item NormalizedDocument, priority float64) *PriorityItem {
	if Q.Len >= Q.Capacity-1 {
		return nil
	}

	Q.Len += 1
	Q.Items[Q.Len] = PriorityItem{
		Value:    item,
		Priority: priority,
		Position: Q.Len,
	}

	it := Q.Len
	parent := int(math.Floor(float64(it) / 2.0))
	for it > 1 && Q.Items[it].Priority < Q.Items[parent].Priority {
		Q.swap(it, parent)
		it = parent
	}

	return &Q.Items[it]
}

func (Q *PriorityQueue) Delete() {
	Q.swap(1, Q.Len)
	Q.Len -= 1
	Q.restore(1, Q.Len)
}

func (Q *PriorityQueue) swap(i, j int) {
	Q.Items[i], Q.Items[j] = Q.Items[j], Q.Items[i]
	Q.Items[i].Position = i
	Q.Items[j].Position = j
}

func (Q *PriorityQueue) restore(i, dim int) {
	min := i
	left := 2 * i
	right := 2*i + 1

	if left <= dim && Q.Items[left].Priority < Q.Items[min].Priority {
		min = left
	}

	if right <= dim && Q.Items[right].Priority < Q.Items[min].Priority {
		min = right
	}

	if i != min {
		Q.swap(i, min)
		Q.restore(min, dim)
	}
}
