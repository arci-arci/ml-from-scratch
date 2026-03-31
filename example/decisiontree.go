package example

import (
	"fmt"
	"ml/classification/decisiontree"
	"ml/common"
	"os"
	"path"
)

func RunDecisionTree() {
	options := decisiontree.DecisionTreeOptions{
		Folders:  []string{"enron1", "enron3", "enron5"},
		Classes:  []string{"ham", "spam"},
		MaxDepth: 25,
		LeafSize: 100,
	}

	model := decisiontree.Train(options)

	testDoc := "enron2"
	testClass := "ham"
	hamFile, err := os.ReadDir(path.Join(testDoc, testClass))
	if err != nil {
		panic(err)
	}

	var tp int
	var fn int

	fmt.Println("Working on 'ham' class")
	maxHam := len(hamFile)

	for _, doc := range hamFile[:maxHam] {
		bow := common.BoW{}
		common.ReadClassDocument(testDoc, testClass, doc.Name(), &bow)
		predictedClass, _ := decisiontree.Fit(&model, &bow)

		if predictedClass == testClass {
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

	maxSpam := len(spamFile)
	for _, doc := range spamFile[:maxSpam] {
		bow := common.BoW{}
		common.ReadClassDocument(testDoc, testClass, doc.Name(), &bow)
		predictedClass, _ := decisiontree.Fit(&model, &bow)

		if predictedClass == testClass {
			tn += 1
		} else {
			fp += 1
		}
	}

	fmt.Printf("\nTest set size 'ham' = %v\n", maxHam)
	fmt.Printf("Test set size 'spam' = %v\n", maxSpam)
	fmt.Printf("True Positive = %v\n", tp)
	fmt.Printf("True Negative = %v\n", tn)
	fmt.Printf("False Negative = %v\n", fn)
	fmt.Printf("False Positive = %v\n", fp)
	fmt.Printf("Recall = %v\n", float64(tp)/float64(tp+fn))
	fmt.Printf("Precision = %v\n", float64(tp)/float64(tp+fp))
	fmt.Printf("Accuracy = %v\n", float64(tp+tn)/float64(tp+tn+fp+fn))
}
