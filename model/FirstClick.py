

from model.Abstract import Abstract

class FirstClick(Abstract):


	def __init__(self):
		pass


	def solve(self, data):

		result = []
		for row in cursor :  
			sourceMediumCampaign = str(row[-1])
			conversionPath = sourceMediumCampaign.split(">") 
			lastSource = conversionPath[-1]
			values = lastSource.split("/")
			lastClick = {
				"Source" : values[0],
				"Medium" : values[1],
				"Campaign" : values[2],
				"totalConversions" : int(row[3]),
				"totalValue" : float(row[4]),
				"Model" : "lastClick"
			}
			result.append(lastClick)


		return result

