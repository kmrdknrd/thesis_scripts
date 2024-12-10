Qualtrics.SurveyEngine.addOnload(function(){	
	const d = new Date();
	var NoTimeDate = d.getFullYear()+'/'+(d.getMonth()+1)+'/'+d.getDate();
	const hours = d.getHours();
	var Date1 = Qualtrics.SurveyEngine.getEmbeddedData("date1");
	var Date2 = Qualtrics.SurveyEngine.getEmbeddedData("date2");
	if ((NoTimeDate == Date2) && (hours >=  10)){
		this.showNextButton();
		jQuery("#QID68").hide();
		jQuery("#QID396").hide();
		jQuery("#QID151").show();
	} else if (NoTimeDate == Date1){
		this.hideNextButton();
		jQuery("#QID151").hide();
		jQuery("#QID396").hide();
		jQuery("#QID68").show();
	} else {
		this.hideNextButton();
		jQuery("#QID151").hide();
		jQuery("#QID68").hide();
		jQuery("#QID396").show();
	}
});