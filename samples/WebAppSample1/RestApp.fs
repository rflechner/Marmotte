module WebAppSample1.RestApp

open System
open System.Threading.Tasks
open Microsoft.AspNetCore.Builder
open Microsoft.AspNetCore.Http
open Marmotte
open Marmotte.RestfulDsl
open Microsoft.AspNetCore.Http.Extensions
open WebAppSample1.Dto

let getUserById (id: int) : Task<UserDto option> =
    Task.FromResult(
        if id = 42 then Some { Id = 42; Name = "Douglas Adams" } else None
    )

// this example demonstrates how to use classic IResult and manually read requests
let getUser (ctx: HttpContext) =
    task {
        match ctx.Request.RouteValues.TryGetValue "id" with
        | true, (:? string as idStr) when idStr |> Int32.TryParse |> fst ->
            let id = int idStr
            let! user = getUserById id
            match user with
            | Some u -> return! Results.Json(u).ExecuteAsync(ctx)
            | None -> return! Results.NotFound("User not found").ExecuteAsync(ctx)
        | _ -> return! Results.BadRequest("Invalid ID").ExecuteAsync(ctx)
    }

// this example demonstrates how to use Restful utilities
let getCustomer (request: CustomerRequestModel) =
    task {
        printfn $"request: %A{request}"
        match request.Id with
        | 42 ->
            let birthDay = DateTime.Today.AddYears(-30) |> DateOnly.FromDateTime
            let name = $"Customer {request.Id}"
            let customer : Customer = { Id=request.Id; Name=name; Birthday=birthDay }
            return Restful.SemanticResult.Ok customer
        | _ -> return Restful.SemanticResult.NotFound(Some "Customer not found")
    }

let searchCustomer (request: SearchCustomerRequestModel) =
    task {
        printfn $"Http context request url: %s{request.HttpContext.Request.GetDisplayUrl()}"
        printfn $"request: %A{request}"
        match request.NamePattern with
        | "joe*" ->
            let birthDay = DateTime.Today.AddYears(-50) |> DateOnly.FromDateTime
            let customer : Customer = { Id=48; Name="Joe Resta"; Birthday=birthDay }
            return Restful.SemanticResult.Ok customer
        | _ -> return Restful.SemanticResult.NotFound(Some "Customer not found")
    }

let run (app: WebApplication) =
    
    // this example demonstrates how to use classic IResult and manually read requests
    app |> Http.get "/users/{id:int}" getUser |> ignore
    
    // If you don't want to use the computation expression, you can uncomment following code

    // app |> Restful.addResource "Customer" (
    //         fun ctx ->
    //             ctx |> Restful.get "/{id:int}" getCustomer id
    //             ctx |> Restful.post "/search" searchCustomer (fun c -> { c with OperationName=Some "SearchCustomer" })
    //         )
    // app.Run()

    // computation expression creates routes
    // this example demonstrates how to use Restful utilities
    restful app {
        resource "Customer"
        get "/{id:int}" getCustomer
        post "/search" searchCustomer
        
        resource "User"
        get "/{id:int}" (
                fun (request:UserRequestModel) ->
                    task {
                        match request.Id with
                        | 42 ->
                            let name = $"User {request.Id}"
                            let user : UserDto = { Id=request.Id; Name=name }
                            return Restful.SemanticResult.Ok user
                        | _ -> return Restful.SemanticResult.NotFound(Some "User not found")
                    }
                )
    }
