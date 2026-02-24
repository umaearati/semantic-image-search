from semantic_image_search.backend.query_translator import translate_query
import time
print("First call:")
print(translate_query("show me a beautiful red sports car"))

print("\nSecond call:")
print(translate_query("show me a beautiful red sports car"))


# long_query = "red sports car " * 50
# print(len(long_query))
# print(translate_query(long_query))

# First call (LLM call expected)
start = time.time()
translate_query("red sports car")
print("First call time:", time.time() - start)

# Second call (cache hit expected)
start = time.time()
translate_query("red sports car")
print("Second call time:", time.time() - start)