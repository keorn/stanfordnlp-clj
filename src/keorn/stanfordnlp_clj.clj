(ns keorn.stanfordnlp-clj
  (:import [java.io StringReader]
           [edu.stanford.nlp.process DocumentPreprocessor]
           [edu.stanford.nlp.tagger.maxent MaxentTagger]
           [edu.stanford.nlp.parser.nndep DependencyParser]))

; Tokenizing
(defn tokenize-sentences [text]
  (mapv vec (iterator-seq (.iterator (DocumentPreprocessor. (StringReader. text))))))

; Tagging
(def ^{:private true} 
  tagger
  (memoize (fn [] (MaxentTagger. MaxentTagger/DEFAULT_JAR_PATH))))

(defn tag-sentence [tokenized-sentence] (.tagSentence (tagger) tokenized-sentence))

; Parsing
(def ^{:private true} 
  dependency-parser
  (memoize (fn [] (DependencyParser/loadFromModelFile DependencyParser/DEFAULT_MODEL))))

(defn dependency-parse [tagged-sentence]
  (.typedDependenciesCCprocessed (.predict (dependency-parser) tagged-sentence)))

(defn triplet [dependency]
  (list (keyword (.. dependency reln toString))
        (.. dependency gov word)
        (.. dependency dep word)))

; Chained
(defn sentence->triplets [sentence]
  (->> sentence tag-sentence dependency-parse (mapv triplet)))
