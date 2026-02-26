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
		probForHam := (float64((*options.hamBoW)[token] + 1)) / (float64(termsForHam + int64(vSize)))
		probForSpam := (float64((*options.spamBoW)[token] + 1)) / (float64(termsForSpam + int64(vSize)))
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

		docInHam += math.Log(tProb.ham)
		docInSpam += math.Log(tProb.spam)
	}

	return ClassificationResult{
		ham:  docInHam + math.Log(options.hamProb),
		spam: docInSpam + math.Log(options.spamProb),
	}
}

func main() {
	hamBoW := BoW{}
	spamBoW := BoW{}
	v := Vocabulary{}
	rootFolder := "enron1"

	err := readClassDocuments(rootFolder, "ham", &hamBoW)
	if err != nil {
		panic(err)
	}

	err = readClassDocuments(rootFolder, "spam", &spamBoW)
	if err != nil {
		panic(err)
	}

	createVocabulary(&hamBoW, &v)
	createVocabulary(&spamBoW, &v)

	hamDocs := getDocAmount(rootFolder, "ham")
	spamDocs := getDocAmount(rootFolder, "spam")
	totalDocs := hamDocs + spamDocs

	hamProb := float64(hamDocs) / float64(totalDocs)
	spamProb := float64(spamDocs) / float64(totalDocs)

	option := BayesOptions{
		hamBoW:     &hamBoW,
		spamBoW:    &spamBoW,
		vocabulary: &v,
	}

	classifier := train(option)
	testBoW := BoW{}
	err = readClassDocument("enron2", "spam", "0026.2001-07-13.SA_and_HP.spam.txt", &testBoW)
	if err != nil {
		panic(err)
	}

	cOption := ClassificationOption{
		hamProb:    hamProb,
		spamProb:   spamProb,
		classifier: &classifier,
		doc:        &testBoW,
	}

	r := classify(cOption)
	fmt.Printf("P(D|ham) = %v\n", r.ham)
	fmt.Printf("P(D|spam) = %v\n", r.spam)
}
