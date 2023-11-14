from flask import Flask,request,redirect,render_template
from src.pipelines.predict_pipeline import customData,predict_pipeline

application=Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/form',methods=['GET','POST'])
def form():
    if(request.method=='GET'):
        return render_template('home.html')
    else:
        AccountWeeks=float(request.form['AccountWeeks'])
        ContractRenewal=float(request.form['ContractRenewal'])
        DataPlan=float(request.form['DataPlan'])
        DataUsage=float(request.form['DataUsage'])
        CustServCalls=float(request.form['CustServCalls'])
        DayMins=float(request.form['DayMins'])
        DayCalls=float(request.form['DayCalls'])
        MonthlyCharge=float(request.form['MonthlyCharge'])
        OverageFee=float(request.form['OverageFee'])
        RoamMins=float(request.form['RoamMins'])


        data=customData(AccountWeeks,ContractRenewal,DataPlan,DataUsage,CustServCalls,DayMins,DayCalls,MonthlyCharge,OverageFee,RoamMins)
        df=data.get_dataframe()
        predict=predict_pipeline(df)
        result=predict.predict_churn()
        if result==0:
            res='not churned'
        else:
            res='churned'
        return render_template('result.html',res=res)
        #return redirect('/result?res=' + res)


if __name__=='__main__':
    app.run(debug=True)