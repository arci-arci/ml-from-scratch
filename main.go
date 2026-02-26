package main

import (
	"fmt"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"strings"
)

type probabilities struct {
	ham  float64
	spam float64
}

type BayesOptions struct {
	hamProb    float64
	spamProb   float64
	hamBoW     *BoW
	spamBoW    *BoW
	vocabulary *Vocabulary
}

type BoW = map[string]int64
type Vocabulary = map[string]int64
type BayesClassifier = map[string]probabilities

func readContent(path string) string {
	data, err := os.ReadFile(path)
	if err != nil {
		panic(err)
	}

	return string(data)
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

		content := readContent(path)
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

		tokens := strings.FieldsSeq(content)

		for token := range tokens {
			(*bow)[token] += 1
		}

		return nil
	})

	return err
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
	classifider := BayesClassifier{}
	termsForHam := calcualteTermsAmount(options.hamBoW)
	termsForSpam := calcualteTermsAmount(options.spamBoW)
	vSize := len(*options.vocabulary)

	for token := range *options.hamBoW {
		probForHam := (float64((*options.hamBoW)[token] + 1)) / (float64(termsForHam + int64(vSize)))
		probForSpam := (float64((*options.spamBoW)[token] + 1)) / (float64(termsForSpam + int64(vSize)))

		tPros := probabilities{
			ham:  probForHam,
			spam: probForSpam,
		}

		classifider[token] = tPros
	}

	return classifider
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
		hamProb:    hamProb,
		spamProb:   spamProb,
		hamBoW:     &hamBoW,
		spamBoW:    &spamBoW,
		vocabulary: &v,
	}

	classifier := train(option)
	for t := range classifier {
		fmt.Printf("%v => %v\n", t, classifier[t])
	}

}
