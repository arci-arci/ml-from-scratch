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
	v := common.Vocabulary{}

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
				common.CreateVocabulary(&bow, &v)
			}
		}
	}

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
	target := normalizePoint(model, p)

	for index, q := range model.Points {
		d := euclideanDistance(&target, q.WBow, model.Vocabulary)
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

func normalizePoint(model *KNNModel, p *common.BoW) WeightedBoW {
	target := WeightedBoW{}
	for token, tf := range *p {
		idf := math.Log2(float64(model.Size) / float64((*model.df)[token]))
		target[token] = float64(tf) * idf
	}

	return target
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
		weightedBoW := WeightedBoW{}
		for token, tf := range *info.Bow {
			idf := math.Log2(float64(size) / float64(df[token]))
			weightedBoW[token] = float64(tf) * idf
		}

		nDoc := NormalizedDocument{
			DocumentName: info.DocumentName,
			Class:        info.Class,
			WBow:         &weightedBoW,
		}

		normalizedDocuments = append(normalizedDocuments, nDoc)
	}

	return normalizedDocuments, df
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
		}
	}

	return foundClass
}

func euclideanDistance(target *WeightedBoW, q *WeightedBoW, v *common.Vocabulary) float64 {
	var distance float64

	for token := range *v {
		x := (*target)[token]
		y := (*q)[token]
		squareValue := math.Pow(float64(x-y), 2)
		distance += squareValue
	}

	return math.Sqrt(distance)
}
