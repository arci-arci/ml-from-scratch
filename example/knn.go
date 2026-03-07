package example

import (
	"fmt"
	"ml/classification/knn"
	"ml/common"
	"os"
	"path"
)

func RunKNN() {
	folders := []string{"enron1"}
	classes := []string{"ham", "spam"}
	model := knn.Train(folders, classes)
	k := 8
	max := 50

	testDoc := "enron2"
	testClass := "ham"
	hamFile, err := os.ReadDir(path.Join(testDoc, testClass))
	if err != nil {
		panic(err)
	}

	var tp int
	var fn int

	fmt.Println("Working on 'ham' class")

	for _, doc := range hamFile[:max] {
		bow := common.BoW{}
		common.ReadClassDocument(testDoc, testClass, doc.Name(), &bow)
		class := knn.Predict(&model, &bow, k)
		fmt.Printf("  %v => %v\n", doc.Name(), class)

		if class == testClass {
			tp += 1
		} else {
			fn += 1
		}
	}

	var fp int
	var tn int
	testClass = "spam"
	fmt.Println("Working on 'spam' class")
	spamFile, err := os.ReadDir(path.Join(testDoc, testClass))
	if err != nil {
		panic(err)
	}

	for _, doc := range spamFile[:max] {
		bow := common.BoW{}
		common.ReadClassDocument(testDoc, testClass, doc.Name(), &bow)
		class := knn.Predict(&model, &bow, k)
		fmt.Printf("  %v => %v\n", doc.Name(), class)

		if class == testClass {
			tn += 1
		} else {
			fp += 1
		}
	}

	fmt.Printf("\nTest set size 'ham' = %v\n", max)
	fmt.Printf("Test set size 'spam' = %v\n", max)
	fmt.Printf("True Positive = %v\n", tp)
	fmt.Printf("True Negative = %v\n", tn)
	fmt.Printf("False Negative = %v\n", fn)
	fmt.Printf("False Positive = %v\n", fp)
	fmt.Printf("Recall = %v\n", float64(tp)/float64(tp+fn))
	fmt.Printf("Precision = %v\n", float64(tp)/float64(tp+fp))
	fmt.Printf("Accuracy = %v\n", float64(tp+tn)/float64(tp+tn+fp+fn))
}
