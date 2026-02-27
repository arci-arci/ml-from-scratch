package main

import (
	"fmt"
	"ml/classification/knn"
	"ml/common"
)

func main() {
	folders := []string{"enron1", "enron4"}
	model := knn.Train(folders)
	bow := common.BoW{}
	common.ReadClassDocument("enron2", "ham", "0020.1999-12-14.kaminski.ham.txt", &bow)
	res := knn.Fit(model, &bow, 3)

	for _, n := range res {
		fmt.Printf("Class => %v\n", n.Class)
		fmt.Printf("Index => %v\n", n.Index)
		fmt.Printf("Distance => %v\n", n.Distance)
		fmt.Printf("-------------------\n")
	}
}
