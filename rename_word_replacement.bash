# Rename (with word replacement)

# Intended use: rename files that contain an undesirable word

for f in *word_desired_to_change*; 
 do mv "$f" "${f/word_desired_to_change/new_word}"; 
done
