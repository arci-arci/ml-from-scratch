package main

import (
	"fmt"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"strings"
)

type BoW = map[string]int64
type Vocabulary = map[string]int64

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

	termsForHam := calcualteTermsAmount(&hamBoW)
	termsForSpam := calcualteTermsAmount(&spamBoW)

	fmt.Printf("P(ham) => %v\n", hamProb)
	fmt.Printf("P(spam) => %v\n", spamProb)
	fmt.Printf("terms inside ham => %v\n", termsForHam)
	fmt.Printf("terms inside spam => %v\n", termsForSpam)
	fmt.Printf("|V| => %v\n", len(v))

	t := "reflected"
	probForHam := (float64(hamBoW[t] + 1)) / (float64(termsForHam + int64(len(v))))
	probForSpam := (float64(spamBoW[t] + 1)) / (float64(termsForHam + int64(len(v))))
	fmt.Printf("P( '%v' | ham ) => %v\n", t, probForHam)
	fmt.Printf("P( '%v' | spam ) => %v\n", t, probForSpam)
}
