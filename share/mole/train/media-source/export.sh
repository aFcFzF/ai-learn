#! /bin/bash
ffmpeg -i ./record.mov -vf "select=not(mod(n\,12))" -vsync vfr ./output/output_%03d.jpg