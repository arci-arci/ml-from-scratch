package knn

import (
	"math"
	"ml/common"
	"os"
	"path"
	"sort"
)

type Point struct {
	Document *common.BoW
	Class    string
}

type KNNModel struct {
	Points []Point
	V      *common.Vocabulary
}

type Neighbor struct {
	Distance float64
	Index    int
	Class    string
}

const initialCapacity int = 10_000

func Train(folders []string) KNNModel {
	db := make([]Point, 0, initialCapacity)
	v := common.Vocabulary{}
	classes := []string{"ham", "spam"}

	for _, folder := range folders {
		for _, class := range classes {
			documents, err := os.ReadDir(path.Join(folder, class))
			if err != nil {
				panic(err)
			}

			for _, document := range documents {
				bow := common.BoW{}
				common.ReadClassDocument(folder, class, document.Name(), &bow)
				db = append(db, Point{Document: &bow, Class: class})
				common.CreateVocabulary(&bow, &v)
			}
		}
	}

	return KNNModel{
		Points: db,
		V:      &v,
	}
}

func Fit(model KNNModel, p *common.BoW, k int) string {
	neighbors := []Neighbor{}

	if k <= 0 {
		panic("k parameter must be greater than 0")
	}

	for index, q := range model.Points {
		d := euclideanDistance(p, q.Document, model.V)
		neighbors = append(neighbors, Neighbor{Distance: d, Index: index, Class: q.Class})
	}

	sort.Slice(neighbors, func(i int, j int) bool {
		return neighbors[i].Distance < neighbors[j].Distance
	})

	return vote(neighbors[:k])
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

func euclideanDistance(p *common.BoW, q *common.BoW, v *common.Vocabulary) float64 {
	var distance float64

	for token := range *v {
		x := (*p)[token]
		y := (*q)[token]
		squareValue := math.Pow(float64(x-y), 2)
		distance += squareValue
	}

	return math.Sqrt(distance)
}
