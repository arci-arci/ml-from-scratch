package main

import (
	"fmt"
	"ml/classification/knn"
	"ml/common"
	"os"
	"path"
)

func main() {
	folders := []string{"enron1"}
	model := knn.Train(folders)
	k := 10
	documents, err := os.ReadDir(path.Join("enron2", "ham"))
	if err != nil {
		panic(err)
	}

	var tp int
	var fn int

	for _, doc := range documents {
		bow := common.BoW{}
		common.ReadClassDocument("enron2", "ham", doc.Name(), &bow)
		class := knn.Fit(model, &bow, k)
		fmt.Printf("Document %v Class => %v\n", doc.Name(), class)

		if class == "ham" {
			tp += 1
		} else {
			fn += 1
		}
	}

	fmt.Printf("Recall = %v\n", float64(tp)/float64(tp+fn))

}
