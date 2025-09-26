sap.ui.require(
    [
        'sap/fe/test/JourneyRunner',
        'sap/btp/ai/log/test/integration/FirstJourney',
		'sap/btp/ai/log/test/integration/pages/MessageTimeAggregatesList',
		'sap/btp/ai/log/test/integration/pages/MessageTimeAggregatesObjectPage',
		'sap/btp/ai/log/test/integration/pages/MessagesObjectPage'
    ],
    function(JourneyRunner, opaJourney, MessageTimeAggregatesList, MessageTimeAggregatesObjectPage, MessagesObjectPage) {
        'use strict';
        var JourneyRunner = new JourneyRunner({
            // start index.html in web folder
            launchUrl: sap.ui.require.toUrl('sap/btp/ai/log') + '/index.html'
        });

       
        JourneyRunner.run(
            {
                pages: { 
					onTheMessageTimeAggregatesList: MessageTimeAggregatesList,
					onTheMessageTimeAggregatesObjectPage: MessageTimeAggregatesObjectPage,
					onTheMessagesObjectPage: MessagesObjectPage
                }
            },
            opaJourney.run
        );
    }
);