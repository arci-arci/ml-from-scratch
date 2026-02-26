package main

import (
	"fmt"
	"io/fs"
	"math"
	"os"
	"path"
	"path/filepath"
	"strings"
)

type probabilities struct {
	ham  float64
	spam float64
}

type ClassificationOption struct {
	hamProb    float64
	spamProb   float64
	classifier *BayesClassifier
	doc        *BoW
}

type BayesOptions struct {
	hamBoW     *BoW
	spamBoW    *BoW
	vocabulary *Vocabulary
}

type ClassificationResult struct {
	ham  float64
	spam float64
}

type BoW = map[string]int64
type Vocabulary = map[string]int64
type BayesClassifier = map[string]probabilities

func readContent(path string) (string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}

	return string(data), nil
}

func cleanFileContent(content string, what string) string {
	return strings.ReplaceAll(content, what, "")
}

func getDocAmount(root string, class string) int {
	docs, dirErr := os.ReadDir(path.Join(root, class))
	if dirErr != nil {
		panic(dirErr)
	}

	return len(docs)
}

func readClassDocuments(root string, class string, bow *BoW) error {
	path := path.Join(root, class)
	err := filepath.WalkDir(path, func(path string, info fs.DirEntry, err error) error {
		if err != nil {
			panic(err)
		}

		if info.IsDir() {
			return nil
		}

		content, err := readContent(path)
		if err != nil {
			panic(err)
		}

		content = cleanFileContent(content, "-")
		content = cleanFileContent(content, "/")
		content = cleanFileContent(content, ";")
		content = cleanFileContent(content, ".")
		content = cleanFileContent(content, ",")
		content = cleanFileContent(content, "@")
		content = cleanFileContent(content, "(")
		content = cleanFileContent(content, ")")
		content = cleanFileContent(content, ":")
		content = cleanFileContent(content, "~")
		content = cleanFileContent(content, "{")
		content = cleanFileContent(content, "}")
		content = cleanFileContent(content, ">")
		content = cleanFileContent(content, "<")

		tokens := strings.FieldsSeq(content)

		for token := range tokens {
			(*bow)[token] += 1
		}

		return nil
	})

	return err
}

func readClassDocument(root string, class string, fileName string, bow *BoW) error {
	path := path.Join(root, class, fileName)
	content, err := readContent(path)
	if err != nil {
		panic(err)
	}

	content = cleanFileContent(content, "-")
	content = cleanFileContent(content, "/")
	content = cleanFileContent(content, ";")
	content = cleanFileContent(content, ".")
	content = cleanFileContent(content, ",")
	content = cleanFileContent(content, "@")
	content = cleanFileContent(content, "(")
	content = cleanFileContent(content, ")")
	content = cleanFileContent(content, ":")
	content = cleanFileContent(content, "~")
	content = cleanFileContent(content, "{")
	content = cleanFileContent(content, "}")
	content = cleanFileContent(content, ">")
	content = cleanFileContent(content, "<")

	tokens := strings.FieldsSeq(content)

	for token := range tokens {
		(*bow)[token] += 1
	}

	return nil
}

func calcualteTermsAmount(bow *BoW) int64 {
	var total int64
	for t := range *bow {
		total += (*bow)[t]
	}

	return total
}

func createVocabulary(bow *BoW, v *Vocabulary) {
	for t := range *bow {
		(*v)[t] += (*bow)[t]
	}
}

func train(options BayesOptions) BayesClassifier {
	classifier := BayesClassifier{}
	termsForHam := calcualteTermsAmount(options.hamBoW)
	termsForSpam := calcualteTermsAmount(options.spamBoW)
	vSize := len(*options.vocabulary)

	for token := range *options.vocabulary {
		// Probability calculated based on Laplace Correction
		probForHam := math.Log(float64((*options.hamBoW)[token]+1)) - math.Log(float64(termsForHam+int64(vSize)))
		probForSpam := math.Log(float64((*options.spamBoW)[token]+1)) - math.Log(float64(termsForSpam+int64(vSize)))
		tProb := probabilities{
			ham:  probForHam,
			spam: probForSpam,
		}

		classifier[token] = tProb
	}

	return classifier
}

func classify(options ClassificationOption) ClassificationResult {
	docInHam := 0.0
	docInSpam := 0.0

	for token := range *options.doc {
		tProb := (*options.classifier)[token]

		// Skipping rare tokens
		if tProb.ham == 0 || tProb.spam == 0 {
			continue
		}

		docInHam += tProb.ham
		docInSpam += tProb.spam
	}

	return ClassificationResult{
		ham:  docInHam + math.Log(options.hamProb),
		spam: docInSpam + math.Log(options.spamProb),
	}
}

func main() {
	v := Vocabulary{}
	folders := []string{"enron1", "enron4", "enron5"}
	hamBoW := BoW{}
	spamBoW := BoW{}
	var totalDocs int
	var totalHamDocs int
	var totalSpamDocs int

	for _, folder := range folders {
		err := readClassDocuments(folder, "ham", &hamBoW)
		if err != nil {
			panic(err)
		}

		err = readClassDocuments(folder, "spam", &spamBoW)
		if err != nil {
			panic(err)
		}

		createVocabulary(&hamBoW, &v)
		createVocabulary(&spamBoW, &v)
		hamDocs := getDocAmount(folder, "ham")
		spamDocs := getDocAmount(folder, "spam")
		totalHamDocs += hamDocs
		totalSpamDocs += spamDocs
	}

	totalDocs = totalHamDocs + totalSpamDocs
	option := BayesOptions{
		hamBoW:     &hamBoW,
		spamBoW:    &spamBoW,
		vocabulary: &v,
	}

	model := train(option)

	hamProb := float64(totalHamDocs) / float64(totalDocs)
	spamProb := float64(totalSpamDocs) / float64(totalDocs)
	hamTestBoW := BoW{}
	spamTestBoW := BoW{}
	testRootFolder := "enron2"

	hamFile, dirErr := os.ReadDir(path.Join(testRootFolder, "ham"))
	if dirErr != nil {
		panic(dirErr)
	}

	var tp int
	var fp int
	var fn int
	var tn int
	for _, doc := range hamFile {
		err := readClassDocument(testRootFolder, "ham", doc.Name(), &hamTestBoW)
		if err != nil {
			panic(err)
		}

		cOption := ClassificationOption{
			hamProb:    hamProb,
			spamProb:   spamProb,
			classifier: &model,
			doc:        &hamTestBoW,
		}

		r := classify(cOption)
		if r.ham > r.spam {
			tp += 1
		} else {
			fn += 1
		}
	}

	spamFile, dirErr := os.ReadDir(path.Join(testRootFolder, "spam"))
	if dirErr != nil {
		panic(dirErr)
	}

	for _, doc := range spamFile {
		err := readClassDocument(testRootFolder, "spam", doc.Name(), &spamTestBoW)
		if err != nil {
			panic(err)
		}

		cOption := ClassificationOption{
			hamProb:    hamProb,
			spamProb:   spamProb,
			classifier: &model,
			doc:        &spamTestBoW,
		}

		r := classify(cOption)
		if r.spam > r.ham {
			tn += 1
		} else {
			fp += 1
		}
	}

	fmt.Printf("True Positive = %v\n", tp)
	fmt.Printf("True Negative = %v\n", tn)
	fmt.Printf("False Negative = %v\n", fn)
	fmt.Printf("False Positive = %v\n", fp)
	fmt.Printf("Recall = %v\n", float64(tp)/float64(tp+fn))
	fmt.Printf("Precision = %v\n", float64(tp)/float64(tp+fp))
	fmt.Printf("Accuracy = %v\n", float64(tp+tn)/float64(tp+tn+fp+fn))

}
