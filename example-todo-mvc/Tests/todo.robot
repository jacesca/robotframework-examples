*** Settings ***
Library    Browser
Library    String
Suite Setup    New Browser    browser=${BROWSER}    headless=${HEADLESS}
Test Setup    New Context    viewport={'width': 1920, 'height': 1080}
Test Teardown    Close Context
Suite Teardown    Close Browser


*** Variables ***
${BROWSER}    chromium
${HEADLESS}    False


*** Test Cases ***
Add Two ToDos And Check Items
    [Documentation]    Checks if ToDos can be added and ToDo count increases
    [Tags]    Add ToDo
    Given ToDo App is open
    When I Add A New ToDo "Learn Robot Framework"
    And I Add A New ToDo "Write Test Cases"
    Then Open ToDos Should Show "2 items left!"
    Take Screenshot    add-todo
    Sleep    3s

Add Two ToDos And Check Wrong Number of Items
    [Documentation]    Checks if ToDos can be added and ToDo count increases
    [Tags]    Add ToDo
    Given ToDo App is open
    When I Add A New ToDo "Learn Robot Framework"
    And I Add A New ToDo "Write Test Cases"
    Then Open ToDos Should Show "2 items left!"

Add ToDo And Mark ToDo
    [Tags]    Mark ToDo
    Given ToDo App is open
    When I Add A New ToDo "Learn Robot Framework"
    And I Mark ToDo "Learn Robot Framework"
    Then Open ToDos Should Show "0 items left!"
    Take Screenshot    mark-todo
    # Sleep    3s

Check If Marked ToDos Are Removed
    [Tags]    Remove ToDo
    Given ToDo App is open
    And I Added Two ToDos
    When I Mark One ToDo
    Then Open ToDos Should Show "1 item left!"
    Take Screenshot    removed-todo
    
Split ToDos
    [Tags]    Add ToDo
    Given ToDo App is open
    When I Add New ToDos "Learn Robot Framework&Write Test Cases&Sleep"
    Then Open ToDos Should Show "3 items left!"
    Take Screenshot    split-todo

Add A Lot Of ToDos
    [Tags]    Massive Add ToDo
    Given ToDo App is open
    When I Add "100" ToDos
    Then Open ToDos Should Show "100 items left!"
    Take Screenshot    massive-add-todo-for

Add A Lot Of Todos With While
    Given ToDo App is open
    When I Add "100" ToDos Using While Loop
    Then Open ToDos Should Show "100 items left!"
    Take Screenshot    massive-add-todo-while

*** Keywords ***
ToDo App is open
    New Page    https://todomvc.com/examples/react/dist/
    

I Add A New ToDo "${todo}"
    Fill Text    .new-todo    ${todo}
    Press Keys    .new-todo    Enter
    

Open ToDos Should Show "${text}"
    Get Text    span.todo-count    ==    ${text}  # span.<class value>


I Mark ToDo "${todo}"
    Click    "${todo}" >> .. >> input.toggle
    # Click:   This is a command provided by the Browser library to simulate a mouse click on a 
    #          specific web element.
    # ${todo}: This is the variable passed to the keyword. It contains the locator or identifier 
    #          of the "ToDo" element you want to interact with. The locator could be an ID, a class, 
    #          or any other attribute that helps the library find the element on the webpage.
    # >> .. >> input.toggle: This is a chaining syntax used in the Browser library to traverse through 
    #          the DOM (Document Object Model).
    #          - The .. notation means "go to the parent element." So, >> .. means "move up to the 
    #            parent element of the current element."
    #          - input.toggle is the target element that is likely a checkbox or a toggle input 
    #            associated with the "ToDo" item. The keyword is instructing the Browser library to 
    #            find this input.toggle element, which is a child of the parent element of ${todo}.


I Added Two ToDos
    I Add A New ToDo "Learn Robot Framework"
    I Add A New ToDo "Write Test Cases"
    

I Mark One ToDo
    Click    li:first-child >> input.toggle
    # Step 1: The li:first-child selector finds the first <li> (list item) element in a list 
    #         (such as an unordered <ul> or ordered <ol> list).
    # Step 2: The >> operator tells the Browser library to look for a child element within that 
    #         first <li> element.
    # Step 3: The input.toggle selector then finds an <input> element with the class toggle 
    #         within that first <li> element.
    # Step 4: The Click command clicks on the input.toggle element found.


I Add New ToDos "${todo}"
    IF  "&" in $todo
        @{todos}    Split String    ${todo}    separator=&
        FOR    ${item}    IN    @{todos}
            Fill Text    .new-todo    ${item}
            Press Keys    .new-todo    Enter
        END
    ELSE
        Fill Text    .new-todo    ${item}
        Press Keys    .new-todo    Enter
    END


I Add "${count}" ToDos
    FOR    ${index}    IN RANGE    ${count}
        I Add A New ToDo "My ToDo Number ${index}"
    END


I Add "${count}" ToDos Using While Loop
    ${x} =    Set Variable    ${0}
    WHILE    ${x} < ${count}
        ${x} =    Evaluate    ${x} + 1
        I Add A New ToDo "My ToDo Number ${x}"
    END
