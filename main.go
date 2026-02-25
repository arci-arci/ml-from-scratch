package main

import (
	"fmt"
	"io/fs"
	"os"
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

func main() {
	bow := BoW{}

	err := filepath.WalkDir("enron1", func(path string, info fs.DirEntry, err error) error {
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

	for token := range bow {
		fmt.Printf("%v => %v\n", token, bow[token])
	}
}
