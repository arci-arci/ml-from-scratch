package common

import (
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"strings"
)

type BoW = map[string]int64
type Vocabulary = map[string]int64

func ReadContent(path string) (string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}

	return string(data), nil
}

func GetDocAmount(root string, class string) int {
	docs, dirErr := os.ReadDir(path.Join(root, class))
	if dirErr != nil {
		panic(dirErr)
	}

	return len(docs)
}

func CalcualteTermsAmount(bow *BoW) int64 {
	var total int64
	for t := range *bow {
		total += (*bow)[t]
	}

	return total
}

func CleanFileContent(content string) string {
	var result strings.Builder

	for i := 0; i < len(content); i++ {
		b := content[i]
		isLowerCase := 'a' <= b && b <= 'z'
		isUpperCase := 'A' <= b && b <= 'Z'
		isDigit := '0' <= b && b <= '9'

		if isLowerCase || isUpperCase || isDigit {
			result.WriteByte(b)
		}
	}

	return result.String()
}

func CreateVocabulary(bow *BoW, v *Vocabulary) {
	for t := range *bow {
		(*v)[t] += (*bow)[t]
	}
}

func ReadClassDocuments(root string, class string, bow *BoW) error {
	path := path.Join(root, class)
	err := filepath.WalkDir(path, func(path string, info fs.DirEntry, err error) error {
		if err != nil {
			panic(err)
		}

		if info.IsDir() {
			return nil
		}

		content, err := ReadContent(path)
		if err != nil {
			panic(err)
		}

		content = strings.ToLower(CleanFileContent(content))
		tokens := strings.FieldsSeq(content)

		for token := range tokens {
			(*bow)[token] += 1
		}

		return nil
	})

	return err
}

func ReadClassDocument(root string, class string, fileName string, bow *BoW) error {
	path := path.Join(root, class, fileName)
	content, err := ReadContent(path)
	if err != nil {
		panic(err)
	}

	content = strings.ToLower(CleanFileContent(content))
	tokens := strings.FieldsSeq(content)

	for token := range tokens {
		(*bow)[token] += 1
	}

	return nil
}
