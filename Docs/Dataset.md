# Dataset

The model looks at one photo at a time and answers one question: is this button defective or not?

## Folder layout

```
Data/
  train/
    ok_front/     photos of good buttons
    def_front/    photos of buttons with a dot or a scratch
  test/
    ok_front/
    def_front/
```

The folder names are fixed. `ok_front` becomes label 0 (OK), `def_front` label 1 (Defective).
A different folder can be used with `python main.py train --data path/to/folder`.

## What is in this repository

Only a handful of photos, just enough to show the layout and to let the tests run:

| Split | OK | Defective |
|-------|----|-----------|
| Train | 2  | 6         |
| Test  | 2  | 4         |

The original set had about 200 photos. They were taken with a phone on a workbench, so the lighting
differs from photo to photo, and most of them were removed to keep the repository small.
With this few photos the numbers printed by `train` say very little about real accuracy. They only
prove the pipeline works.

## Preparing your own photos

- Use the same camera, distance and lighting for every photo. This matters more than the number of photos.
- Aim for at least 100 photos per class. Keep the classes roughly balanced; the training also weights
  the rarer class higher, but that does not replace real examples.
- Test photos must be different photos from the training ones, not crops of the same shot.
- JPEG, PNG and BMP files work. Photos are converted to grayscale and resized to 300 x 300.
- A quarter of the training photos is held back for validation while training.
