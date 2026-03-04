package knn

import (
	"math"
	"ml/common"
	"os"
	"path"
	"sort"
	"strings"
)

type Point struct {
	Document *common.BoW
	Class    string
}

type KNNModel struct {
	Points     []NormalizedDocument
	Vocabulary *common.Vocabulary
	Size       int
	df         *DocumentFrequency
}

type Neighbor struct {
	Distance     float64
	Index        int
	Class        string
	DocumentName string
}

type DocumentData struct {
	DocumentName string
	Bow          *common.BoW
	Class        string
}

type NormalizedDocument struct {
	DocumentName string
	WBow         *WeightedBoW
	Class        string
}

type DocumentFrequency = map[string]int
type WeightedBoW = map[string]float64

func Train(folders []string, classes []string) KNNModel {
	db, df := normalize(folders, classes)
	v := defineVocabulary(folders, classes)
	_ = CreateBallTree(&v, db)

	return KNNModel{
		Points:     db,
		Vocabulary: &v,
		Size:       len(db),
		df:         &df,
	}
}

func Fit(model *KNNModel, p *common.BoW, k int) string {
	if k <= 0 {
		panic("k parameter must be greater than 0")
	}

	neighbors := []Neighbor{}
	target := getWeithedBoW(p, model.df, model.Size)

	for index, q := range model.Points {
		d := EuclideanDistance(&target, q.WBow, model.Vocabulary)
		neighbor := Neighbor{
			Distance: d, Index: index,
			Class: q.Class, DocumentName: q.DocumentName,
		}

		neighbors = append(neighbors, neighbor)
	}

	sort.Slice(neighbors, func(i int, j int) bool {
		return neighbors[i].Distance < neighbors[j].Distance
	})

	return vote(neighbors[:k])
}

func defineVocabulary(folders []string, classes []string) common.Vocabulary {
	v := common.Vocabulary{}

	for _, folder := range folders {
		for _, class := range classes {
			bow := common.BoW{}
			common.ReadClassDocuments(folder, class, &bow)
			common.CreateVocabulary(&bow, &v)
		}
	}

	return common.ClearVocabulary(&v)
}

func normalize(folders []string, classes []string) ([]NormalizedDocument, DocumentFrequency) {
	size, err := getCollectionSize(folders, classes)
	df := DocumentFrequency{}
	documentsData := make([]DocumentData, 0, size)
	normalizedDocuments := make([]NormalizedDocument, 0, size)

	if err != nil {
		panic(err)
	}

	for _, folder := range folders {
		for _, class := range classes {
			documents, err := os.ReadDir(path.Join(folder, class))
			if err != nil {
				panic(err)
			}

			for _, document := range documents {
				if strings.ToLower(document.Name()) == "summary.txt" {
					continue
				}

				bow := common.BoW{}
				common.ReadClassDocument(folder, class, document.Name(), &bow)
				addToDocFrequency(&df, &bow)
				documentsData = append(documentsData, DocumentData{DocumentName: document.Name(), Bow: &bow, Class: class})
			}
		}
	}

	for _, info := range documentsData {
		weightedBoW := getWeithedBoW(info.Bow, &df, size)
		nDoc := NormalizedDocument{
			DocumentName: info.DocumentName,
			Class:        info.Class,
			WBow:         &weightedBoW,
		}

		normalizedDocuments = append(normalizedDocuments, nDoc)
	}

	return normalizedDocuments, df
}

func calculateTermFrequency(bow *common.BoW) WeightedBoW {
	var totalFrequencies int64
	weightedBoW := WeightedBoW{}

	for token := range *bow {
		totalFrequencies += (*bow)[token]
	}

	for token := range *bow {
		weightedBoW[token] = float64((*bow)[token]) / float64(totalFrequencies)
	}

	return weightedBoW
}

func getWeithedBoW(bow *common.BoW, df *DocumentFrequency, size int) WeightedBoW {
	weightedBoW := WeightedBoW{}
	tfBoW := calculateTermFrequency(bow)

	for token, tf := range tfBoW {
		// In case a term is not present in a document, I add 1 on both of the term
		idf := math.Log10(float64(1+size) / float64(1+(*df)[token]))
		weightedBoW[token] = tf * idf
	}

	return weightedBoW
}

func getCollectionSize(folders []string, classes []string) (int, error) {
	size := 0
	for _, folder := range folders {
		for _, class := range classes {
			documents, err := os.ReadDir(path.Join(folder, class))
			if err != nil {
				return -1, err
			}

			size += len(documents)
		}
	}

	return size, nil
}

func addToDocFrequency(df *DocumentFrequency, bow *common.BoW) {
	for token := range *bow {
		numDocs, ok := (*df)[token]

		if ok {
			(*df)[token] = numDocs + 1
		} else {
			(*df)[token] = 1
		}
	}
}

func vote(neighbors []Neighbor) string {
	classes := []string{"ham", "spam"}
	foundClass := classes[0]
	maxFrequency := -1

	for _, class := range classes {
		frequency := 0

		for _, n := range neighbors {
			if class == n.Class {
				frequency += 1
			}
		}

		if frequency > maxFrequency {
			foundClass = class
			maxFrequency = frequency
		}
	}

	return foundClass
}

func EuclideanDistance(target *WeightedBoW, q *WeightedBoW, v *common.Vocabulary) float64 {
	var distance float64

	for token := range *v {
		x := (*target)[token]
		y := (*q)[token]
		squareValue := math.Pow(float64(x-y), 2)
		distance += squareValue
	}

	return math.Sqrt(distance)
}
