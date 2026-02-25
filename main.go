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

func main() {
	bow := BoW{}
	rootFolder := "enron1"

	err := filepath.WalkDir(rootFolder, func(path string, info fs.DirEntry, err error) error {
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
			bow[token] += 1
		}

		return nil
	})

	if err != nil {
		panic(err)
	}

	hamDocs := getDocAmount(rootFolder, "ham")
	spamDocs := getDocAmount(rootFolder, "spam")
	totalDocs := hamDocs + spamDocs

	fmt.Printf("P(ham) => %v\n", float64(hamDocs)/float64(totalDocs))
	fmt.Printf("P(spam) => %v\n", float64(spamDocs)/float64(totalDocs))
}
