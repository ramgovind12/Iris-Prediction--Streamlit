from django.http import JsonResponse
from django.shortcuts import render

# Create your views here.
def predict(request):
    return render(request, 'predict.html')

def predict_changes(request):
    if request.POST.get('action') == 'post':
        sepal_length = float(request.POST.get('sepal_length'))
        sepal_width = float(request.POST.get('sepal_width'))
        petal_length = float(request.POST.get('petal_length'))
        petal_width = float(request.POST.get('petal_width'))

        model = pd.read_pickle('predict')
        result = model.predict([[sepal_length,sepal_width,petal_length,petal_width]])

        classification = result[0]

        return JsonResponse({'result': classification,'sepal_length':sepal_length,'sepal_width':sepal_width,
        'petal_length':petal_length,'petal_width':petal_width},safe = False)