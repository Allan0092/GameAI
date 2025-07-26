using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class chief_NPC_interaction : MonoBehaviour
{
    public string interactButtonName = "talk_text"; // Set this to the name of your interact button GameObject
    public float interactionRange = 1f; // Adjust this value based on your desired range

    private GameObject interactButton; // Reference to the UI interact button GameObject


    // Dialogue GameObject references
    public string textBubbleName = "text_bubble sprite_0";
    private GameObject text_bubble;
    private GameObject chief_dialogue_one;
    private GameObject chief_dialogue_two;
    private GameObject chief_dialogue_three;
    private GameObject chief_dialogue_four;
    private GameObject chief_dialogue_five;
    private GameObject chief_dialogue_six;

    // Boolean required to give the player the bow, and open the gates to exit the village, which can be seen
    // in the 'changing_sprite_image.cs' C# scripts
    public bool completedinteraction;

    // Start is called before the first frame update
    void Start()
    {
        text_bubble = GameObject.Find(textBubbleName);
        chief_dialogue_one = GameObject.Find("chiefdialogueOne");
        chief_dialogue_two = GameObject.Find("chiefdialogueTwo");
        chief_dialogue_three = GameObject.Find("chiefdialogueThree");
        chief_dialogue_four = GameObject.Find("chiefdialogueFour");
        chief_dialogue_five = GameObject.Find("chiefdialogueFive");
        chief_dialogue_six = GameObject.Find("chiefdialogueSix");

        text_bubble.SetActive(false);
        chief_dialogue_one.SetActive(false);
        chief_dialogue_two.SetActive(false);
        chief_dialogue_three.SetActive(false);
        chief_dialogue_four.SetActive(false);
        chief_dialogue_five.SetActive(false);
        chief_dialogue_six.SetActive(false);

        completedinteraction = false;

        // Find the interact button GameObject using its name
        interactButton = GameObject.Find(interactButtonName);

        if (interactButton == null)
        {
            Debug.LogError("Interact button not found. Make sure to set the correct name.");
        }
    }


    // Update is called once per frame
    void Update()
    {
        CheckPlayerDistance();
    }


    void CheckPlayerDistance()
    {
        GameObject player = GameObject.FindGameObjectWithTag("Player");

        character_movement characterMovementVariable = player.GetComponent<character_movement>();
        character_jump_movement characterJumpMovementVariable = player.GetComponent<character_jump_movement>();

        if (player != null)
        {
            float distance = Vector2.Distance(transform.position, player.transform.position);

            if (distance <= interactionRange)
            {
                ShowInteractButton();


                // Dialogue features:
                
                // Allowing interaction if 
                if (Input.GetKeyDown(KeyCode.I))
                {


                    // Locking character movement feature:
                    if (characterMovementVariable != null)
                    {
                        // Halt the character's movement
                        characterMovementVariable.ToggleMovement(false);
                    }

                    // Locking character jump movement feature:
                    if (characterJumpMovementVariable != null)
                    {
                        // Halt the character's movement
                        characterJumpMovementVariable.ToggleJumpMovement(false);
                    }


                    if (!text_bubble.activeSelf && !chief_dialogue_one.activeSelf && !chief_dialogue_two.activeSelf && !chief_dialogue_three.activeSelf && !chief_dialogue_four.activeSelf && !chief_dialogue_five.activeSelf && !chief_dialogue_six.activeSelf)
                    {
                        StartDialogue();
                    }
                    else if (text_bubble.activeSelf && chief_dialogue_one.activeSelf && !chief_dialogue_two.activeSelf && !chief_dialogue_three.activeSelf && !chief_dialogue_four.activeSelf && !chief_dialogue_five.activeSelf && !chief_dialogue_six.activeSelf)
                    {
                        SecondDialogue();
                    }
                    else if (text_bubble.activeSelf && !chief_dialogue_one.activeSelf && chief_dialogue_two.activeSelf && !chief_dialogue_three.activeSelf && !chief_dialogue_four.activeSelf && !chief_dialogue_five.activeSelf && !chief_dialogue_six.activeSelf)
                    {
                        ThirdDialogue();
                    }
                    else if (text_bubble.activeSelf && !chief_dialogue_one.activeSelf && !chief_dialogue_two.activeSelf && chief_dialogue_three.activeSelf && !chief_dialogue_four.activeSelf && !chief_dialogue_five.activeSelf && !chief_dialogue_six.activeSelf)
                    {
                        FourthDialogue();
                    }
                    else if (text_bubble.activeSelf && !chief_dialogue_one.activeSelf && !chief_dialogue_two.activeSelf && !chief_dialogue_three.activeSelf && chief_dialogue_four.activeSelf && !chief_dialogue_five.activeSelf && !chief_dialogue_six.activeSelf)
                    {
                        FifthDialogue();
                    }
                    else if (text_bubble.activeSelf && !chief_dialogue_one.activeSelf && !chief_dialogue_two.activeSelf && !chief_dialogue_three.activeSelf && !chief_dialogue_four.activeSelf && chief_dialogue_five.activeSelf && !chief_dialogue_six.activeSelf)
                    {
                        SixthDialogue();
                    }
                    else if (text_bubble.activeSelf && !chief_dialogue_one.activeSelf && !chief_dialogue_two.activeSelf && !chief_dialogue_three.activeSelf && !chief_dialogue_four.activeSelf && !chief_dialogue_five.activeSelf && chief_dialogue_six.activeSelf)
                    {
                        EndDialogue();
                        characterMovementVariable.ToggleMovement(true);
                        characterJumpMovementVariable.ToggleJumpMovement(true);

                        completedinteraction = true;
                    }
                    
                }


            }
            else
            {
                HideInteractButton();
            }
        }
    }

    void ShowInteractButton()
    {
        interactButton.SetActive(true);
    }

    void HideInteractButton()
    {

        interactButton.SetActive(false);

    }

    // Call this method for the first dialogue
    void StartDialogue()
    {
        text_bubble.SetActive(true);
        chief_dialogue_one.SetActive(true);
    }

    // Call this method when the first dialogue is complete
    void SecondDialogue()
    {
        chief_dialogue_one.SetActive(false);
        chief_dialogue_two.SetActive(true);
    }

    // Call this method when the second dialogue is complete
    void ThirdDialogue()
    {
        chief_dialogue_two.SetActive(false);
        chief_dialogue_three.SetActive(true);
    }

    // Call this method when the third dialogue is complete
    void FourthDialogue()
    {
        chief_dialogue_three.SetActive(false);
        chief_dialogue_four.SetActive(true);
    }

    // Call this method when the fourth dialogue is complete
    void FifthDialogue()
    {
        chief_dialogue_four.SetActive(false);
        chief_dialogue_five.SetActive(true);
    }

    // Call this method when the fifth dialogue is complete
    void SixthDialogue()
    {
        chief_dialogue_five.SetActive(false);
        chief_dialogue_six.SetActive(true);
    }

    // Call this method when the sixth dialogue is complete
    void EndDialogue()
    {
        chief_dialogue_six.SetActive(false);
        text_bubble.SetActive(false);
    }
}