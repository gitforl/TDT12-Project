import keras_nlp
import numpy as np
import pathlib
import random
import tensorflow as tf
import random
import requests

from tensorflow import keras
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab



BATCH_SIZE = 64
EPOCHS = 15
MAX_SEQUENCE_LENGTH = 40
MODERN_VOCAB_SIZE = 15000
ORIGINAL_VOCAB_SIZE = 15000

EMBED_DIM = 256
INTERMEDIATE_DIM = 2048
NUM_HEADS = 8

base_url = "https://raw.githubusercontent.com/cocoxu/Shakespeare/master/data/align/plays/merged/"

paired_plays = [
    ("antony-and-cleopatra_modern.snt.aligned","antony-and-cleopatra_original.snt.aligned"),
    ("asyoulikeit_modern.snt.aligned","asyoulikeit_original.snt.aligned"),
    ("errors_modern.snt.aligned","errors_original.snt.aligned"),
    ("hamlet_modern.snt.aligned","hamlet_original.snt.aligned"),
    ("henryv_modern.snt.aligned","henryv_original.snt.aligned"),
    ("juliuscaesar_modern.snt.aligned","juliuscaesar_original.snt.aligned"),
    ("lear_modern.snt.aligned","lear_original.snt.aligned"),
    ("macbeth_modern.snt.aligned","macbeth_original.snt.aligned"),
    ("merchant_modern.snt.aligned","merchant_original.snt.aligned"),
    ("msnd_modern.snt.aligned","msnd_original.snt.aligned"),
    ("muchado_modern.snt.aligned","muchado_original.snt.aligned"),
    ("othello_modern.snt.aligned","othello_original.snt.aligned"),
    ("richardiii_modern.snt.aligned","richardiii_original.snt.aligned"),
    ("romeojuliet_modern.snt.aligned","romeojuliet_original.snt.aligned"),
    ("shrew_modern.snt.aligned","shrew_original.snt.aligned"),
    ("tempest_modern.snt.aligned","tempest_original.snt.aligned"),
    ("twelfthnight_modern.snt.aligned","twelfthnight_original.snt.aligned"),
]






def getLinePairs(paired_plays: list[tuple[str, str]]):

    paired_lines = []

    for pair in paired_plays:

        request1 = requests.get(base_url + pair[0])
        request2 = requests.get(base_url + pair[1])

        modern_content = request1.content.decode("utf-8").split('\n')
        original_content = request2.content.decode("utf-8").split('\n')


        for lines in zip(modern_content, original_content):
            modern = lines[0].lower()
            original = lines[1].lower()
            paired_lines.append((modern, original))

    return paired_lines

paired_lines = getLinePairs(paired_plays)
# paired_lines = random.shuffle(paired_lines)

num_val_samples = int(len(paired_lines) * 0.15)
num_train_samples = len(paired_lines) - 2 * num_val_samples

train_pairs = paired_lines[: num_train_samples]
val_pairs = paired_lines[num_train_samples : num_train_samples + num_val_samples]
test_pairs = paired_lines[num_train_samples + num_val_samples :]

# print(paired_lines[0:5])
# print(len(paired_lines))

print(f"{len(train_pairs)} train pairs")
print(f"{len(val_pairs)} val pairs")
print(f"{len(test_pairs)} test pairs")


def train_word_piece(text_samples, vocab_size, reserved_tokens):
    bert_vocab_args = dict(
        vocab_size=vocab_size,
        reserved_tokens=reserved_tokens,
        bert_tokenizer_params={"lower_case": True},
    )

    word_piece_ds = tf.data.Dataset.from_tensor_slices(text_samples)
    vocab = bert_vocab.bert_vocab_from_dataset(
        word_piece_ds.batch(1000).prefetch(2), **bert_vocab_args
    )

    return vocab

reserved_tokens = ["[PAD]", "[UNK]", "[START]", "[END]"]

modern_samples = [text_pair[0] for text_pair in paired_lines]
modern_vocab = train_word_piece(modern_samples, MODERN_VOCAB_SIZE, reserved_tokens)

original_samples = [text_pair[1] for text_pair in paired_lines]
original_vocab = train_word_piece(original_samples, ORIGINAL_VOCAB_SIZE, reserved_tokens)

# print("Modern Tokens: ", modern_vocab[100:110])
# print("Original Tokens: ", original_vocab[100:110])

modern_tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    vocabulary=modern_vocab, lowercase=False
)

original_tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    vocabulary=original_vocab, lowercase=False
)


# modern_input_example = paired_lines[0][0]
# modern_tokens_example = modern_tokenizer.tokenize(modern_input_example)
# print("Modern sentence: ", modern_input_example)
# print("Tokens: ", modern_tokens_example)
# print("Recovered text after detokenizing: ", modern_tokenizer.detokenize(modern_tokens_example))

# print()

# original_input_example = paired_lines[0][1]
# original_tokens_example = original_tokenizer.tokenize(original_input_example)
# print("Original sentence: ", original_input_example)
# print("Tokens: ", original_tokens_example)
# print("Recovered text after detokenizing: ", original_tokenizer.detokenize(original_tokens_example))

def preprocess_batch(modern, original):
    batch_size = tf.shape(original)[0]

    modern = modern_tokenizer(modern)
    original = modern_tokenizer(original)

    modern_start_end_packer = keras_nlp.layers.StartEndPacker(
        sequence_length=MAX_SEQUENCE_LENGTH,
        pad_value=modern_tokenizer.token_to_id("[PAD]"),
    )
    modern = modern_start_end_packer(modern)

    original_start_end_packer = keras_nlp.layers.StartEndPacker(
        sequence_length= MAX_SEQUENCE_LENGTH + 1,
        start_value= original_tokenizer.token_to_id("[START]"),
        end_value= original_tokenizer.token_to_id("[END]"),
        pad_value= original_tokenizer.token_to_id("[PAD]"),
    )
    original = original_start_end_packer(original)

    return (
        {
            "encoder_inputs": modern,
            "decoder_inputs": original[:,:-1],
        },
        original[:, 1:],
    )

def make_dataset(pairs):
    modern_texts, original_texts = zip(*pairs)
    modern_texts = list(modern_texts)
    original_texts = list(original_texts)
    dataset = tf.data.Dataset.from_tensor_slices((modern_texts, original_texts))
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.map(preprocess_batch, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.shuffle(2048).prefetch(16).cache()

train_ds = make_dataset(train_pairs)
val_ds = make_dataset(val_pairs)


# for inputs, targets in train_ds.take(1):
#     print(f'inputs["encoder_inputs"].shape: {inputs["encoder_inputs"].shape}')
#     print(f'inputs["decoder_inputs"].shape: {inputs["decoder_inputs"].shape}')
#     print(f"targets.shape: {targets.shape}")

#Encoder


encoder_inputs = keras.Input(shape=(None,), dtype="int64", name="encoder_inputs")


x = keras_nlp.layers.TokenAndPositionEmbedding(
    vocabulary_size=MODERN_VOCAB_SIZE,
    sequence_length=MAX_SEQUENCE_LENGTH,
    embedding_dim=EMBED_DIM,
    mask_zero=True,
)(encoder_inputs)

encoder_outputs = keras_nlp.layers.TransformerEncoder(
    intermediate_dim=INTERMEDIATE_DIM, num_heads=NUM_HEADS
)(inputs=x)
encoder = keras.Model(encoder_inputs, encoder_outputs, name="encoder")

#Decoder

decoder_inputs = keras.Input(shape=(None,), dtype="int64", name="decoder_inputs")
encoded_seq_inputs = keras.Input(shape=(None, EMBED_DIM), name="decoder_state_inputs")

x = keras_nlp.layers.TokenAndPositionEmbedding(
    vocabulary_size=ORIGINAL_VOCAB_SIZE,
    sequence_length=MAX_SEQUENCE_LENGTH,
    embedding_dim=EMBED_DIM,
    mask_zero=True,
)(decoder_inputs)

x = keras_nlp.layers.TransformerDecoder(
    intermediate_dim=INTERMEDIATE_DIM, num_heads=NUM_HEADS
)(decoder_sequence=x, encoder_sequence=encoded_seq_inputs)
x = keras.layers.Dropout(0.5)(x)
decoder_outputs = keras.layers.Dense(ORIGINAL_VOCAB_SIZE, activation="softmax")(x)
decoder = keras.Model(
    [
        decoder_inputs,
        encoded_seq_inputs,
    ],
    decoder_outputs,
    name="decoder",
)
decoder_outputs = decoder([decoder_inputs, encoder_outputs])

transformer = keras.Model(
    [encoder_inputs, decoder_inputs],
    decoder_outputs,
    name="transformer",
)

"Training"

transformer.summary()
transformer.compile(
    "rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
transformer.fit(train_ds, epochs=EPOCHS, validation_data=val_ds)

# transformer.save("Shakespear30Epochs")

# transformer = keras.models.load_model("Shakespear")

def decode_sentences(input_sentences):
    bacth_size = tf.shape(input_sentences)[0]

    encoder_input_tokens = modern_tokenizer(input_sentences).to_tensor(
        shape=(None, MAX_SEQUENCE_LENGTH)
    )

    def token_probability_fn(decoder_input_tokens):
        return transformer([encoder_input_tokens, decoder_input_tokens])[:, -1,:]

    prompt = tf.fill((bacth_size, 1), original_tokenizer.token_to_id("[START]"))

    generated_tokens = keras_nlp.utils.greedy_search(
        token_probability_fn,
        prompt,
        max_length=40,
        end_token_id=original_tokenizer.token_to_id("[END]"),
    )
    generated_sentences = original_tokenizer.detokenize(generated_tokens)
    return generated_sentences

test_modern_texts = [pair[0] for pair in test_pairs]
for i in range(10):
    input_sentence = random.choice(test_modern_texts)
    translated = decode_sentences(tf.constant([input_sentence]))
    translated = translated.numpy()[0].decode("utf-8")
    translated = (
        translated.replace("[PAD]", "")
        .replace("[START]", "")
        .replace("[END]", "")
        .strip()
    )
    print(f"** Example {i} **")
    print(input_sentence)
    print(translated)
    print()

rouge_1 = keras_nlp.metrics.RougeN(order=1)
rouge_2 = keras_nlp.metrics.RougeN(order=2)

for test_pair in test_pairs[:30]:
    input_sentence = test_pair[0]
    reference_sentence = test_pair[1]

    translated_sentence = decode_sentences(tf.constant([input_sentence]))
    translated_sentence = translated_sentence.numpy()[0].decode("utf-8")
    translated_sentence = (
        translated_sentence.replace("[PAD]", "")
        .replace("[START]", "")
        .replace("[END]", "")
        .strip()
    )

    rouge_1(reference_sentence, translated_sentence)
    rouge_2(reference_sentence, translated_sentence)

print("ROUGE-1 Score: ", rouge_1.result())
print("ROUGE-2 Score: ", rouge_2.result())
