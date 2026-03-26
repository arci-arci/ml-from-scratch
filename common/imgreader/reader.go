package imgreader

import (
	"image"
	"image/png"
	"io"
	"math"
	"os"
)

type Pixel struct {
	R int
	G int
	B int
	A int
}

const conversionUnit uint32 = 257

func GetPixelFromImage(path string) ([]Pixel, error) {
	image.RegisterFormat("png", "png", png.Decode, png.DecodeConfig)
	file, err := os.Open(path)

	if err != nil {
		return nil, err
	}

	defer file.Close()
	pixels, err := getPixels(file)

	if err != nil {
		return nil, err
	}

	return pixels, nil
}

func getPixels(file io.Reader) ([]Pixel, error) {
	img, _, err := image.Decode(file)

	if err != nil {
		return nil, err
	}

	bounds := img.Bounds()
	width, height := bounds.Max.X, bounds.Max.Y
	var pixels []Pixel

	for y := range height {
		var row []Pixel

		for x := range width {
			r, g, b, a := img.At(x, y).RGBA()
			row = append(row, rgbaToPixel(r, g, b, a))
		}

		pixels = append(pixels, row...)
	}

	return pixels, nil
}

func rgbaToPixel(r, g, b, a uint32) Pixel {
	// Because we are dealing with 16-bit alpha-premultiplied color
	// in the range [0, 65535], we convert each compoent's value to a
	// 8-bit alpha-premultiplied color in the range [0, 255]
	// by dividing each component by 0x101 (257)
	//
	// https://go.dev/blog/image#colors-and-color-models

	return Pixel{
		R: int(r / conversionUnit),
		G: int(g / conversionUnit),
		B: int(b / conversionUnit),
		A: int(a / conversionUnit),
	}
}

func (a Pixel) Sum(b Pixel) Pixel {
	return Pixel{
		R: a.R + b.R,
		G: a.G + b.G,
		B: a.B + b.B,
		A: a.A + b.A,
	}
}

func (a Pixel) Div(scalar int) Pixel {
	return Pixel{
		R: a.R / scalar,
		G: a.G / scalar,
		B: a.B / scalar,
		A: a.A / scalar,
	}
}

func (a Pixel) Distance(b Pixel) float64 {
	sum := math.Pow((float64(a.R) - float64(b.R)), 2)
	sum += math.Pow((float64(a.G) - float64(b.G)), 2)
	sum += math.Pow((float64(a.B) - float64(b.B)), 2)
	sum += math.Pow((float64(a.A) - float64(b.A)), 2)

	return math.Sqrt(sum)
}
