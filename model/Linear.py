

from model.Abstract import Abstract

class Linear(Abstract):


	def __init__(self):
		pass


	def solve(self, data):

		for row in data :
			sourceMediumCampaign = str(row[-1])
			conversionPath = sourceMediumCampaign.split(">") 
			for conversion in conversionPath:
				values = conversion.split("/")
				linear = {
					"Source" : values[0],
					"Medium" : values[1],
					"Campaign" : values[2],
					"totalConversions" : int(row[3])/len(conversionPath),
					"totalValue" : float(row[4])/len(conversionPath),
					"Model" : "linear"
				}
				result.append(linear)


		return result

