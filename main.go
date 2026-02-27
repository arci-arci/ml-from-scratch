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
	k := 100
	common.ReadClassDocument("enron2", "ham", "0020.1999-12-14.kaminski.ham.txt", &bow)
	// common.ReadClassDocument("enron2", "spam", "0026.2001-07-13.SA_and_HP.spam.txt", &bow)
	class := knn.Fit(model, &bow, k)
	fmt.Printf("Class => %v\n", class)
}
