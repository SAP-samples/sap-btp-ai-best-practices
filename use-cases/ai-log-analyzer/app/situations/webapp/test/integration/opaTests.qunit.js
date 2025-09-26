sap.ui.require(
    [
        'sap/fe/test/JourneyRunner',
        'sap/btp/ai/situations/test/integration/FirstJourney',
		'sap/btp/ai/situations/test/integration/pages/SituationsList',
		'sap/btp/ai/situations/test/integration/pages/SituationsObjectPage',
		'sap/btp/ai/situations/test/integration/pages/MessageTimeAggregatesObjectPage'
    ],
    function(JourneyRunner, opaJourney, SituationsList, SituationsObjectPage, MessageTimeAggregatesObjectPage) {
        'use strict';
        var JourneyRunner = new JourneyRunner({
            // start index.html in web folder
            launchUrl: sap.ui.require.toUrl('sap/btp/ai/situations') + '/index.html'
        });

       
        JourneyRunner.run(
            {
                pages: { 
					onTheSituationsList: SituationsList,
					onTheSituationsObjectPage: SituationsObjectPage,
					onTheMessageTimeAggregatesObjectPage: MessageTimeAggregatesObjectPage
                }
            },
            opaJourney.run
        );
    }
);