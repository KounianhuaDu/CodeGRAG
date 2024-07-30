from dataloaders.datadealer import DataDealer


total = 0
datadealer = DataDealer(dataset="CodeForce", split="test")
for id, problem in datadealer.iter_test_data():
    total += len(problem["example_testcases"])

print(total)
