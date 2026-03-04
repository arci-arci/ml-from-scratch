package example

import (
	"fmt"
	bayes "ml/classification/bayes"
	"ml/common"
	"os"
	"path"
)

func RunBayes() {
	v := common.Vocabulary{}
	max := 400
	folders := []string{"enron1"}
	hamBoW := common.BoW{}
	spamBoW := common.BoW{}
	var totalDocs int
	var totalHamDocs int
	var totalSpamDocs int

	for _, folder := range folders {
		err := common.ReadClassDocuments(folder, "ham", &hamBoW)
		if err != nil {
			panic(err)
		}

		err = common.ReadClassDocuments(folder, "spam", &spamBoW)
		if err != nil {
			panic(err)
		}

		common.CreateVocabulary(&hamBoW, &v)
		common.CreateVocabulary(&spamBoW, &v)
		hamDocs := common.GetDocAmount(folder, "ham")
		spamDocs := common.GetDocAmount(folder, "spam")
		totalHamDocs += hamDocs
		totalSpamDocs += spamDocs
	}

	totalDocs = totalHamDocs + totalSpamDocs
	bayesOptions := bayes.BayesOptions{
		HamBoW:     &hamBoW,
		SpamBoW:    &spamBoW,
		Vocabulary: &v,
	}

	model := bayes.Train(bayesOptions)
	hamProb := float64(totalHamDocs) / float64(totalDocs)
	spamProb := float64(totalSpamDocs) / float64(totalDocs)
	testRootFolder := "enron2"

	hamFile, dirErr := os.ReadDir(path.Join(testRootFolder, "ham"))
	if dirErr != nil {
		panic(dirErr)
	}

	var tp int
	var fp int
	var fn int
	var tn int

	for _, doc := range hamFile[:max] {
		hamTestBoW := common.BoW{}
		err := common.ReadClassDocument(testRootFolder, "ham", doc.Name(), &hamTestBoW)
		if err != nil {
			panic(err)
		}

		cOptions := bayes.ClassificationOptions{
			HamProb:  hamProb,
			SpamProb: spamProb,
			Model:    &model,
			Doc:      &hamTestBoW,
		}

		r := bayes.Fit(cOptions)
		if r.Ham > r.Spam {
			tp += 1
		} else {
			fn += 1
		}
	}

	spamFile, dirErr := os.ReadDir(path.Join(testRootFolder, "spam"))
	if dirErr != nil {
		panic(dirErr)
	}

	for _, doc := range spamFile[:max] {
		spamTestBoW := common.BoW{}
		err := common.ReadClassDocument(testRootFolder, "spam", doc.Name(), &spamTestBoW)
		if err != nil {
			panic(err)
		}

		cOptions := bayes.ClassificationOptions{
			HamProb:  hamProb,
			SpamProb: spamProb,
			Model:    &model,
			Doc:      &spamTestBoW,
		}

		r := bayes.Fit(cOptions)
		if r.Spam > r.Ham {
			tn += 1
		} else {
			fp += 1
		}
	}

	fmt.Printf("Test set size 'ham' = %v\n", len(hamFile))
	fmt.Printf("Test set size 'spam' = %v\n", len(spamFile))
	fmt.Printf("True Positive = %v\n", tp)
	fmt.Printf("True Negative = %v\n", tn)
	fmt.Printf("False Negative = %v\n", fn)
	fmt.Printf("False Positive = %v\n", fp)
	fmt.Printf("Recall = %v\n", float64(tp)/float64(tp+fn))
	fmt.Printf("Precision = %v\n", float64(tp)/float64(tp+fp))
	fmt.Printf("Accuracy = %v\n", float64(tp+tn)/float64(tp+tn+fp+fn))

}
