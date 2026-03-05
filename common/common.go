package common

import (
	"io/fs"
	"iter"
	"os"
	"path"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"unicode"
)

type BoW = map[string]int64
type Vocabulary = map[string]int64

const MIN_WORD_LEN int = 3
const AVG_TOKEN_AMOUNT int = 83

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

	for _, r := range content {
		if unicode.IsLower(r) || unicode.IsUpper(r) ||
			unicode.IsNumber(r) || unicode.IsSpace(r) {
			result.WriteRune(r)
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

		if strings.ToLower(info.Name()) == "summary.txt" {
			return nil
		}

		content, err := ReadContent(path)
		if err != nil {
			panic(err)
		}

		content = strings.ToLower(CleanFileContent(content))
		tokens := strings.FieldsSeq(content)
		fillBoW(tokens, bow)

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
	fillBoW(tokens, bow)
	return nil
}

func fillBoW(tokens iter.Seq[string], bow *BoW) {
	for token := range tokens {
		if _, err := strconv.Atoi(token); err == nil {
			// Skip numbers
			continue
		}

		if len(token) <= MIN_WORD_LEN {
			// Skip words with three or less characters
			continue
		}

		if isAStopWord(token) {
			// Skip stop words
			continue
		}

		(*bow)[token] += 1
	}
}

func isAStopWord(token string) bool {
	stopWords := getStopWords()
	return slices.Contains(stopWords, token)
}

func getStopWords() []string {
	return []string{
		"i", "me", "my", "myself",
		"we", "our", "ours", "ourselves",
		"you", "your", "yours", "yourself",
		"yourselves", "he", "him", "his",
		"himself", "she", "her", "hers",
		"herself", "it", "its", "itself",
		"they", "them", "their", "theirs",
		"themselves", "what", "which", "who",
		"whom", "this", "that", "these", "those",
		"am", "is", "are", "was", "were", "be",
		"been", "being", "have", "has", "had",
		"having", "do", "does", "did", "doing",
		"a", "an", "the", "and", "but", "if",
		"or", "because", "as", "until", "while",
		"of", "at", "by", "for", "with", "about",
		"against", "between", "into", "through",
		"during", "before", "after", "above",
		"below", "to", "from", "up", "down",
		"in", "out", "on", "off", "over", "under",
		"again", "further", "then", "once", "here",
		"there", "when", "where", "why", "how", "all",
		"any", "both", "each", "few", "more", "most",
		"other", "some", "such", "no", "nor", "not", "only",
		"own", "same", "so", "than", "too", "very", "s",
		"t", "can", "will", "just", "don", "should", "now",
	}
}
